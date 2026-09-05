# RAG Company Knowledge Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant
information from company documents and uses a selected AI model to generate
grounded answers.

## Project architecture

```text
User → Streamlit frontend → FastAPI backend → retrieval → LLM → grounded answer
```

Documents are loaded from `data/`, split into chunks, embedded with the local
`all-MiniLM-L6-v2` model, and persisted in the configured vector database.

## Run locally

From the project root, install dependencies and start the two applications:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload
.\venv\Scripts\streamlit.exe run frontend/app.py
```

Copy `.env.example` to `.env` and select the model provider without changing
Python code:

```text
LLM_PROVIDER=ollama   # or openai
VECTOR_DB=chroma      # or faiss
```

Ollama defaults to `qwen2.5-coder:7b`; make sure that model is available in
your local Ollama installation. For OpenAI, set `OPENAI_API_KEY` in `.env`.
The `.env` file is ignored by Git.

## Vector-store options

The application stores normalized embeddings and converts each store's squared
L2 distance into the displayed similarity score `1 / (1 + distance)`. Higher
scores mean closer chunks; this is a comparable retrieval score, not a model
confidence value.

| Store | Setup and persistence | Speed / best use case |
| --- | --- | --- |
| Chroma (default) | No extra package beyond `requirements.txt`; persists in `chroma_db/` and keeps the existing project data. | Convenient document metadata and local development. Good default for this chatbot. |
| FAISS | Set `VECTOR_DB=faiss`; persists its index and chunk metadata in `faiss_db/`. | Typically very fast local similarity search with a lightweight index. Best for a simple local, read-heavy knowledge base; metadata filtering is intentionally minimal here. |
