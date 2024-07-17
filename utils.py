import requests
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def initialize_image_embeddings(gmap_id, urls, image_collection):
    for i, url in enumerate(urls):
        try:
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
                embeddings=[image_features],
                ids=[f"{gmap_id}_img_{i}"],
                metadatas=[{"url": url, "gmap_id": gmap_id}],
            )
        except Exception as e:
            print(f"Error processing image {url}: {e}")


def image_text_matching(text, gmap_id, image_collection):
    # Encode the input text
    text_embedding = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = (
            model.encode_text(text_embedding).cpu().numpy().flatten().tolist()
        )

    # Query Chroma for the closest matches
    results = image_collection.query(
        query_embeddings=[text_features], where={"gmap_id": gmap_id}, n_results=3
    )

    # Extract the top k image URLs
    top_images = [result["url"] for result in results["metadatas"][0]]
    print("top_images:", top_images)

    return top_images
