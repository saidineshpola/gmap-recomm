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
from utils import initialize_image_embeddings, image_text_matching, safe_len
from typing import List, Dict, Any
import json
import hashlib

# Ensure cache directory exists
cache_dir = "query_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

app = FastAPI()

# Set up the embedding model
embed_model = HuggingFaceEmbedding(
    model_name="./bge-small-en-v1.5"
)  # BAAI/bge-small-en-v1.5
Settings.embed_model = embed_model

# Global variables to store loaded data
index = None
gmap_id_to_data = {}
user_reviews = {}
business_reviews = {}
business_images = {}
vector_index = None
keyword_index = None

# Initialize Chroma client and collection for images
chroma_client = chromadb.PersistentClient(path="./chroma_db_images")
image_collection = chroma_client.get_or_create_collection(
    "image_embeddings", metadata={"hnsw:space": "cosine"}
)


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

                if gmap_id not in business_reviews:
                    business_reviews[gmap_id] = []
                business_reviews[gmap_id].append(review)
                user_reviews[user_id].append(review)

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


class FollowUpQuery(BaseModel):
    query: str
    previous_results: Dict[str, Any]
    conversation_history: List[Dict[str, str]]


def get_context(results, user_id):
    context = "Based on the query, here are some relevant queries from user:\n\n"
    for i, result in enumerate(results, 1):
        context += f"{i}. {result['text']}\n"
        context += f"   Details: {json.dumps(result['data'], indent=2)}\n"

        business_id = result["data"]["gmap_id"]

        if result["user_reviews"]:
            context += "   User's past reviews:\n"
            for review in result["user_reviews"]:
                context += (
                    f"   - Rating: {review['rating']}, Review: {review['text']}\n"
                )
        context += "\n"
    return context


hasher = hashlib.sha256()


@app.post("/query")
async def query_endpoint(input: str, user_id: str, conversation_id: int = None):
    if vector_index is None:
        raise HTTPException(status_code=500, detail="Index Data not loaded")

    # Generate a hash for the query
    query_hash = hashlib.sha256(f"{input}:{user_id}".encode()).hexdigest()

    # Check if cached response exists
    cache_file = os.path.join(cache_dir, f"{query_hash}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # If not cached, proceed with regular code
    query_engine = vector_index.as_retriever()
    response = query_engine.retrieve(input)
    results = []
    for r in response:
        business_id = r.metadata["businessId"]
        result = {
            "text": r.text,
            "data": gmap_id_to_data[business_id],
            "user_reviews": [],
            "images": list(business_images.get(business_id, set())),
        }
        if user_id and user_id in user_reviews:
            for review in user_reviews[user_id]:
                result["user_reviews"].append(review)

        existing_images = image_collection.get(where={"gmap_id": business_id})
        if not existing_images["ids"]:
            initialize_image_embeddings(business_id, result["images"], image_collection)
            print(f"Image embeddings initialized for business {business_id}")

        if result["images"]:
            top_images = image_text_matching(input, business_id, image_collection)
            result["top_images"] = top_images
            print("Top-k Images Generated")
        else:
            result["top_images"] = []

        results.append(result)

    context = get_context(results, user_id)

    messages = [
        {
            "role": "system",
            "content": "You are a location-based recommendation assistant giving highly recommended places based on context and user's past reviews. Use the provided information to answer the user's query and Only respond with answer nothing else.",
        }
    ]
    conversations = messages

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser Query: {input}\n Answer USER query only nothing else.",
        }
    )

    ollama_response = ollama.chat(model="llama3", messages=messages)

    response_data = {
        "query_hash": query_hash,
        "response": ollama_response["message"]["content"],
        "results": results,
        "conversation_history": [
            {
                "role": "system",
                "content": "You are a location-based recommendation assistant giving highly recommended places based on context and user's past reviews. Use the provided information to answer the user's query.",
            },
            {"role": "user", "content": input},
            {"role": "assistant", "content": ollama_response["message"]["content"]},
        ],
    }

    # Save the response to cache
    with open(cache_file, "w") as f:
        json.dump(response_data, f)

    return response_data


@app.post("/query_business")
async def query_with_reviews_endpoint(
    input: str, user_id: str, conversation_id: int = None
):
    if vector_index is None:
        raise HTTPException(status_code=500, detail="Index Data not loaded")

    # Generate a hash for the query
    query_hash = hashlib.sha256(f"{input}:{user_id}:with_reviews".encode()).hexdigest()

    # Check if cached response exists
    cache_file = os.path.join(cache_dir, f"{query_hash}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # If not cached, proceed with regular code
    query_engine = vector_index.as_retriever()
    response = query_engine.retrieve(input)
    results = []
    for r in response:
        business_id = r.metadata["businessId"]
        result = {
            "text": r.text,
            "data": gmap_id_to_data[business_id],
            "user_reviews": [],
            "business_reviews": [],
            "images": list(business_images.get(business_id, set())),
        }
        if user_id and user_id in user_reviews:
            for review in user_reviews[user_id]:
                result["user_reviews"].append(review)

        # Add top 10 business reviews
        if business_id in business_reviews:
            # TODO get the top-10 reviews related to the query
            # option1 : vectorDB to get the top-10 related reviews
            # option2 : get the top-10 reviews from the business_reviews
            result["business_reviews"] = sorted(
                business_reviews[business_id],
                key=lambda x: safe_len(x["text"]),
                reverse=True,
            )[:10]
            print(f"Business reviews for {business_id} added")
            # print(result["business_reviews"])

        existing_images = image_collection.get(where={"gmap_id": business_id})
        if not existing_images["ids"]:
            initialize_image_embeddings(business_id, result["images"], image_collection)
            print(f"Image embeddings initialized for business {business_id}")

        if result["images"]:
            top_images = image_text_matching(input, business_id, image_collection)
            result["top_images"] = top_images
            print("Top-k Images Generated")
        else:
            result["top_images"] = []

        results.append(result)

    context = get_context_with_reviews(results, user_id)

    messages = [
        {
            "role": "system",
            "content": "You are a location-based recommendation assistant giving highly recommended places based on context, user's past reviews, and top business reviews. Use the provided information to answer the user's query and Only respond with answer nothing else.",
        }
    ]
    conversations = messages

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser Query: {input}\n Answer USER query only nothing else.",
        }
    )

    ollama_response = ollama.chat(model="gmap_recomm_llama3", messages=messages)

    response_data = {
        "query_hash": query_hash,
        "response": ollama_response["message"]["content"],
        "results": results,
        "conversation_history": [
            {
                "role": "system",
                "content": "You are a location-based recommendation assistant giving highly recommended places based on context, user's past reviews, and top business reviews. Use the provided information to answer the user's query.",
            },
            {"role": "user", "content": input},
            {"role": "assistant", "content": ollama_response["message"]["content"]},
        ],
    }

    # Save the response to cache
    with open(cache_file, "w") as f:
        json.dump(response_data, f)

    return response_data


def get_context_with_reviews(results, user_id):
    context = "Based on the query, here are some relevant queries from user:\n\n"
    for i, result in enumerate(results, 1):
        context += f"{i}. {result['text']}\n"
        context += f"   Details: {json.dumps(result['data'], indent=2)}\n"

        business_id = result["data"]["gmap_id"]

        if result["user_reviews"]:
            context += "   User's past reviews:\n"
            for review in result["user_reviews"]:
                context += (
                    f"   - Rating: {review['rating']}, Review: {review['text']}\n"
                )

        if result["business_reviews"]:
            context += "  Business reviews:\n"
            for review in result["business_reviews"]:
                context += (
                    f"   - Rating: {review['rating']}, Review: {review['text']}\n"
                )

        context += "\n"
    return context


@app.post("/follow_up_query")
async def follow_up_query_endpoint(query: FollowUpQuery):
    if vector_index is None:
        raise HTTPException(status_code=500, detail="Index Data not loaded")

    # Use the previous results instead of performing a new search
    results = query.previous_results["results"]

    # Update top images for the first result (assuming it's the most relevant)
    if results and results[0]["images"]:
        business_id = results[0]["data"]["gmap_id"]
        top_images = image_text_matching(query.query, business_id, image_collection)
        results[0]["top_images"] = top_images
        print("Top-k Images Updated")

    context = get_context(results, query.previous_results.get("user_id", ""))

    messages = [
        {
            "role": "system",
            "content": "You are a location-based recommendation assistant giving highly recommended places based on context and user's past reviews. Use the provided information to answer the user's query.",
        }
    ]

    messages.extend(query.conversation_history)

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser Query: {query.query}\n Answer USER query only nothing else.",
        }
    )

    ollama_response = ollama.chat(model="llama3", messages=messages)

    return {
        "response": ollama_response["message"]["content"],
        "results": results,
        "conversation_history": query.conversation_history
        + [
            {"role": "user", "content": query.query},
            {"role": "assistant", "content": ollama_response["message"]["content"]},
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
