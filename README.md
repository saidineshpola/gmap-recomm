# Local Business Assistant

This is a Streamlit app that uses a language model to assist with local business queries. It connects to a local API endpoint and provides responses based on the user's input.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Datasets](#datasets)
- [Initialize Backend](#initialize-backend)
- [Run the App](#run-the-app)
- [TODO](#todo)

## Features

- Ask questions about local businesses
- Get basic details about businesses
- Follow up on previous queries

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
