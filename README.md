# Enhancing Local Business Intelligence: Beyond Google Maps

This project is a Streamlit app that uses a large language model from Ollama to assist with local business queries and recommendations. It connects to a local API endpoint and provides responses based on the user's input.

## Table of Contents
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Ollama Model](#ollama-model)
- [Initialize Backend](#initialize-backend)
- [Run the App](#run-the-app)
- [Demo](#demo)
- [RAFT](#raft)
- [TODO](#todo)
- [Blog](#blog)

## Architecture

- FastAPI Backend: Handles data processing, retrieval, and integration with the LLM.
- Vector Store: Utilizes Chroma DB for efficient similarity search.
- Embedding Model: Employs HuggingFace's BAAI/bge-small-en-v1.5 for text embeddings.
- Large Language Model: Uses a fine-tuned LLaMA3-8B model for natural language understanding and generation.
- Image Embedding and Matching: Incorporates image-based search for enhanced recommendations using CLIP's features.
- Streamlit Frontend: Provides an intuitive user interface for interacting with the system.

![Architecture](https://drive.google.com/uc?export=view&id=1MYzk_n1co_9LXjJWU52CF9GCZIifig1l)

## Key Features

1. **Personalized Recommendations**
   - Takes into account the user's past reviews and preferences
   - Stores and retrieves user-specific review data
   - Incorporates user reviews into the context provided to the LLM

2. **Multi-modal Search**
   - Incorporates image data for enhanced search capabilities
   - Performs image-text matching to find visually relevant results
   - Presents top matching images alongside text recommendations

3. **Conversational Interface**
   - Chat-like interface for natural language queries
   - Detailed responses from the LLM
   - View relevant images and business details
   - Engage in follow-up questions for deeper exploration

4. **Efficient Data Retrieval**
   - Uses Chroma DB as a vector store for both text and image embeddings
   - Implements semantic search using the BAAI/bge-small-en-v1.5 embedding model
   - Uses CLIP embedding with Chroma DB for image retrievals

## Folder Structure

```
project_root/
│
├── assets/
│   └── demo.gif
│
├── datasets/
│   └── google-local-dataset/
│
├── models/
│   └── gguf_model/
│       └── Modelfile
│
├── notebooks/
│   └── finetuning_notebook.ipynb
│
├── raft/
│   └── README.md
│
├── fast_api.py # Backend
|
├── streamlit_demo.py # frontend
│
├── requirements.txt
└── README.md
```

## Installation

To run this app, you need to have Python and Streamlit installed on your machine. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Datasets

Download the datasets from [google-local-dataset](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/#subsets) and save them inside the `datasets` directory.

## Ollama Model

- You can use the model directly for inference if it fits the system; otherwise, you have to use [unsloth](https://github.com/unslothai/unsloth) model conversion to convert it to GGUF format.
- Use the following code from unsloth to convert it to the GGUF model with q4_k_m quantization:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("../checkpoint_xx")
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method = "q4_k_m")
```

Convert the [finetuned model](https://drive.google.com/drive/folders/1VGyEen8RjsoP-OJL6MowOqUuWIkNQH7i) from the [finetuning notebook](notebooks/finetuning_notebook.ipynb) to Ollama using the following command:

```bash
ollama create gmap_recomm_llama3 -f ./gguf_model/Modelfile
```

> [!NOTE]
> Update the path of the file in [Modelfile](https://drive.google.com/drive/folders/1VGyEen8RjsoP-OJL6MowOqUuWIkNQH7i)

## Initialize Backend

```bash
python src/backend/fast_api.py
```

## Run the App

```bash
streamlit run src/frontend/streamlit_demo.py
```

## Demo

![Local Business Assistant Demo](assets/demo.gif)

*Demo of the Local Business Assistant in action*

## RAFT

RAFT is a recipe for adapting LLMs to domain-specific RAG. For information about the RAFT technique, please refer to the README file in the `raft` directory.

## TODO

- [x] Improve error handling and user feedback
- [x] Optimize database queries for faster responses
- [x] Implement caching mechanism for frequent queries
- [x] Generate RAFT dataset using LLaMA3
- [x] Finetune the LLaMA3/local LLM on the new dataset created 
- [ ] Replace Chroma DB retriever with BM25 from llama-index (package installation issue)
- [ ] Add unit tests for backend functions
- [ ] Integrate with more data sources for comprehensive information
- [ ] Implement a feedback system for users to rate responses

## Blog

The content is also explained briefly in my [blog post](https://www.hackster.io/r-bot/enhancing-local-business-intelligence-beyond-google-map-46939f).
