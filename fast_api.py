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
from cachetools import TTLCache, cached

app = FastAPI()

# Set up the embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Global variables to store loaded data
index = None
gmap_id_to_data = {}
user_reviews = {}

# Cache for user conversations
conversation_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour


def load_data():
    global index, gmap_id_to_data, user_reviews

    def read_lines_as_json(path):
        lines_as_json = []
        with gzip.open(path, "rt", encoding="utf-8") as g:
            for line in g:
                json_line = json.loads(line)
                lines_as_json.append(json_line)
        return lines_as_json

    def read_user_reviews(path):
        user_reviews = {}
        with gzip.open(path, "rt", encoding="utf-8") as g:
            for line in g:
                review = json.loads(line)
                user_id = review["user_id"]
                if user_id not in user_reviews:
                    user_reviews[user_id] = []
                user_reviews[user_id].append(review)
        return user_reviews

    user_reviews_path = "./datasets/indiana/review-Indiana_10.json.gz"
    user_reviews = read_user_reviews(user_reviews_path)

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
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )


@app.on_event("startup")
async def startup_event():
    load_data()
    print("Data loaded")


class Query(BaseModel):
    query: str
    user_id: str
    conversation_id: str = None


@cached(cache=conversation_cache)
def get_conversation_history(conversation_id):
    return []


@app.post("/query")
async def query_endpoint(query: Query):
    if index is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    query_engine = index.as_retriever()
    response = query_engine.retrieve(query.query)
    results = []
    for r in response:
        business_id = r.metadata["businessId"]
        result = {
            "text": r.text,
            "data": gmap_id_to_data[business_id],
            "user_reviews": [],
        }
        if query.user_id and query.user_id in user_reviews:
            for review in user_reviews[query.user_id]:
                result["user_reviews"].append(review)
        results.append(result)

    context = "Based on the query, here are some relevant queries from user:\n\n"
    for i, result in enumerate(results, 1):
        context += f"{i}. {result['text']}\n"
        context += f"   Details: {json.dumps(result['data'], indent=2)}\n"
        if result["user_reviews"]:
            context += "   User's past reviews:\n"
            for review in result["user_reviews"]:
                context += (
                    f"   - Rating: {review['rating']}, Review: {review['text']}\n"
                )
        context += "\n"

    # Get conversation history
    conversation_history = get_conversation_history(query.conversation_id)

    # Prepare messages for ollama
    messages = [
        {
            "role": "system",
            "content": "You are a location-based recommendation assistant giving highly recommended places based on context and user's past reviews. Use the provided information to answer the user's query.",
        }
    ]

    # Add conversation history
    messages.extend(conversation_history)

    # Add current context and query
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser Query: {query.query}\n Answer USER query only nothing else.",
        }
    )

    ollama_response = ollama.chat(model="llama3", messages=messages)

    # Update conversation history
    conversation_history.append({"role": "user", "content": query.query})
    conversation_history.append(
        {"role": "assistant", "content": ollama_response["message"]["content"]}
    )
    conversation_cache[query.conversation_id] = conversation_history

    return {"response": ollama_response["message"]["content"], "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
