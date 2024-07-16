import requests
import clip
import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
import chromadb

from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize Chroma client and collection
chroma_client = chromadb.Client()
image_collection = chroma_client.get_or_create_collection(
    "image", metadata={"hnsw:space": "cosine"}
)
img_urls = {}


# TODO: Save all the Image embeddings before and use them in the image_text_matching function
def initialize_image_embeddings(image_urls):

    for i, url in enumerate(image_urls):
        if url in img_urls:
            print(f"URL already exists in the collection: {url}")
            continue

        try:
            img_urls[url] = i
            image = (
                preprocess(Image.open(requests.get(url, stream=True).raw))
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                image_features = (
                    model.encode_image(image).cpu().numpy().flatten().tolist()
                )

            image_collection.add(
                embeddings=[image_features], ids=[f"img_{i}"], metadatas=[{"url": url}]
            )
        except Exception as e:
            print(f"Error processing image {url}: {e}")


def image_text_matching(image_urls, text, first_time=True):
    # Initialize image embeddings if not already done
    if first_time or image_collection.count() == 0:
        initialize_image_embeddings(image_urls)

    # Encode the input text
    text_embedding = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = (
            model.encode_text(text_embedding).cpu().numpy().flatten().tolist()
        )

    # Query Chroma for the closest matches
    results = image_collection.query(query_embeddings=[text_features], n_results=3)

    # Extract the top k image URLs
    top_images = [result["url"] for result in results["metadatas"][0]]
    print("top_images:", top_images)

    return top_images
