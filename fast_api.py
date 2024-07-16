import os
import gzip
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from utils import image_text_matching


app = FastAPI()

# Set up the embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Global variables to store loaded data
index = None
gmap_id_to_data = {}
user_reviews = {}
business_reviews = {}
business_images = {}
vector_index = None
keyword_index = None
# Dictionary to store conversation history and context
conversation_data = {}


def load_data():
    global vector_index, keyword_index, gmap_id_to_data, user_reviews, business_reviews, business_images

    def read_lines_as_json(path):
        lines_as_json = []
        with gzip.open(path, "rt", encoding="utf-8") as g:
            for line in g:
                json_line = json.loads(line)
                lines_as_json.append(json_line)
        return lines_as_json

    def read_user_reviews(path):
        user_reviews = {}
        business_reviews = {}
        business_images = {}
        with gzip.open(path, "rt", encoding="utf-8") as g:
            for line in g:
                review = json.loads(line)
                user_id = review["user_id"]
                gmap_id = review["gmap_id"]

                if user_id not in user_reviews:
                    user_reviews[user_id] = []
                user_reviews[user_id].append(review)

                # if gmap_id not in business_reviews:
                #     business_reviews[gmap_id] = []
                # business_reviews[gmap_id].append(review)

                if gmap_id not in business_images:
                    business_images[gmap_id] = set()
                if review["pics"]:
                    for pic in review["pics"]:
                        business_images[gmap_id].add(
                            pic["url"][0].replace("=w150-h150-k-no-p", "")
                        )

        return user_reviews, business_reviews, business_images

    user_reviews_path = "./datasets/indiana/review-Indiana_10.json.gz"
    user_reviews, business_reviews, business_images = read_user_reviews(
        user_reviews_path
    )

    path = "datasets/indiana/meta-Indiana.json.gz"
    lines_as_strings = read_lines_as_json(path)

    documents = []
    for t in lines_as_strings:
        if "gmap_id" in t:
            address_or_name = t["address"] if t["address"] else t.get("name", "")
            if address_or_name:
                documents.append(
                    Document(
                        text=address_or_name, metadata={"businessId": t["gmap_id"]}
                    )
                )
            gmap_id_to_data[t["gmap_id"]] = t

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if os.path.exists("datasets/indiana/chroma_index"):
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    else:
        vector_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )


@app.on_event("startup")
async def startup_event():
    load_data()
    print("Data loaded")


class Query(BaseModel):
    query: str
    user_id: str
    conversation_id: str


def get_context(results, user_id):
    context = "Based on the query, here are some relevant queries from user:\n\n"
    for i, result in enumerate(results, 1):
        context += f"{i}. {result['text']}\n"
        context += f"   Details: {json.dumps(result['data'], indent=2)}\n"

        business_id = result["data"]["gmap_id"]
        # if business_id in business_reviews:
        #     context += "   Business reviews:\n"
        #     for review in business_reviews[business_id][
        #         :5
        #     ]:  # Limit to 5 reviews for brevity
        #         context += (
        #             f"   - Rating: {review['rating']}, Review: {review['text']}\n"
        #         )

        if result["user_reviews"]:
            context += "   User's past reviews:\n"
            for review in result["user_reviews"]:
                context += (
                    f"   - Rating: {review['rating']}, Review: {review['text']}\n"
                )
        context += "\n"
    return context


@app.post("/query")
async def query_endpoint(query: Query):
    if vector_index is None:
        raise HTTPException(status_code=500, detail="Index Data not loaded")

    if query.conversation_id not in conversation_data:

        # New conversation, perform semantic search
        query_engine = vector_index.as_retriever()
        response = query_engine.retrieve(query.query)
        results = []
        for r in response:
            business_id = r.metadata["businessId"]
            result = {
                "text": r.text,
                "data": gmap_id_to_data[business_id],
                "user_reviews": [],
                # "business_reviews": business_reviews.get(business_id, []),
                "images": list(business_images.get(business_id, set())),
            }
            if query.user_id and query.user_id in user_reviews:
                for review in user_reviews[query.user_id]:
                    result["user_reviews"].append(review)

            # Get top 2 closest images
            if result["images"]:
                top_images = image_text_matching(result["images"], query.query)
                result["top_images"] = top_images
                print("Top-k Images Generated")
            else:
                result["top_images"] = []

            results.append(result)

        context = get_context(results, query.user_id)

        conversation_data[query.conversation_id] = {
            "history": [],
            "context": context,
            "results": results,
        }
    else:
        if conversation_data[query.conversation_id]["results"][0]["images"]:
            top_images = image_text_matching(
                conversation_data[query.conversation_id]["results"][0]["images"],
                query.query,
            )
            conversation_data[query.conversation_id]["results"][0][
                "images"
            ] = top_images
            print("Top-k Images Generated")
        # Existing conversation, load context and results
        context = conversation_data[query.conversation_id]["context"]
        results = conversation_data[query.conversation_id]["results"]

    # Prepare messages for ollama
    messages = [
        {
            "role": "system",
            "content": "You are a location-based recommendation assistant giving highly recommended places based on context and user's past reviews. Use the provided information to answer the user's query.",
        }
    ]

    # Add conversation history
    messages.extend(conversation_data[query.conversation_id]["history"])

    # Add current context and query
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser Query: {query.query}\n Answer USER query only nothing else.",
        }
    )

    ollama_response = ollama.chat(model="llama3", messages=messages)

    # Update conversation history
    conversation_data[query.conversation_id]["history"].append(
        {"role": "user", "content": query.query}
    )
    conversation_data[query.conversation_id]["history"].append(
        {"role": "assistant", "content": ollama_response["message"]["content"]}
    )

    return {
        "response": ollama_response["message"]["content"],
        "results": results,
        "conversation_history": conversation_data[query.conversation_id]["history"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
