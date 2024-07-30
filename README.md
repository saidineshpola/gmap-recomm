# Local Business Assistant

This is a Streamlit app that uses a language model to assist with local business queries. It connects to a local API endpoint and provides responses based on the user's input.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Demo](#demo)
- [Installation](#installation)
- [Datasets](#datasets)
- [Initialize Backend](#initialize-backend)
- [Run the App](#run-the-app)
- [TODO](#todo)

## Features

- Ask questions about local businesses
- Get basic details about businesses
- Follow up on previous queries

## Architecture

- FastAPI Backend: Handles data processing, retrieval, and integration with the LLM.
- Vector Store: Utilizes Chroma DB for efficient similarity search.
- Embedding Model: Employs HuggingFace's BAAI/bge-small-en-v1.5 for text embeddings.
- Large Language Model: Uses a fine-tuned LLaMA3-8B model for natural language understanding and generation.
- Image Embedding and Matching: Incorporates image-based search for enhanced recommendations using CLIP's features.
- Streamlit Frontend: Provides an intuitive user interface for interacting with the system.

## Demo

![Local Business Assistant Demo](assets/demo.gif)

*Demo of the Local Business Assistant in action*

<!-- <video width="640" height="360" controls>
  <source src="assets/demo.webm" type="video/mp4">
  Your browser does not support the video tag.
</video> -->

## Installation

To run this app, you need to have Python and Streamlit installed on your machine. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Datasets

Download the datasets from [google-local-dataset](https:/g) and save it inside the datasets directory.

## Initialize Backend

```bash
python fast_api.py
```

## Run the App

```bash
streamlit run streamlit_demo.py
```

## TODO

- [x] Improve error handling and user feedback
- [x] Optimize database queries for faster responses
- [x] Implement caching mechanism for frequent queries
- [ ] Replace chromaDB retriever with BM25 from llama-index(package installation issue)
- [ ] Add unit tests for backend functions
- [ ] Integrate with more data sources for comprehensive information
- [ ] Implement a feedback system for users to rate responses