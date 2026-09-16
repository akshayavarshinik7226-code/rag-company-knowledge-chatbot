# RAG Company Knowledge Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant information from company documents and uses a selected AI model to generate grounded answers.

## Project architecture

```text
User → Streamlit frontend → FastAPI backend → retrieval → LLM → grounded answer
```

Documents are loaded from `data/`, split into chunks, embedded with the local `all-MiniLM-L6-v2` model, and persisted in the configured vector database.

## Run locally

### 1. Clone the repository

```powershell
git clone https://github.com/akshayavarshinik7226-code/rag-company-knowledge-chatbot.git
cd rag-company-knowledge-chatbot
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use the virtual environment's Python executable directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Configure environment variables

Create `.env` from the provided example:

```powershell
Copy-Item .env.example .env
```

The default configuration uses Ollama and ChromaDB:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434
VECTOR_DB=chroma
```

For OpenAI, change `LLM_PROVIDER` to `openai` and provide `OPENAI_API_KEY` in `.env`.

### 6. Prepare the local Ollama model

Make sure Ollama is installed and the configured model is available:

```powershell
ollama pull qwen2.5-coder:7b
```

If `ollama` is not available on the PATH, start Ollama from its installed location and keep the default base URL:

```text
http://127.0.0.1:11434
```

### 7. Start the FastAPI backend

From the project root:

```powershell
python -m uvicorn backend.main:app --reload
```

The API should be available at:

```text
http://127.0.0.1:8000
```

Verify the backend:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/ping
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

### 8. Start the Streamlit frontend

Open a second PowerShell terminal, activate the environment, and run:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Streamlit will display the local URL for the chatbot in the terminal.

## Git and local files

The repository intentionally excludes local environment and generated files from Git. These include:

* `.env`
* `venv/`
* Python cache files
* `.pytest_cache/`
* `chroma_db/`

The `.env.example` file is committed as a safe configuration template.

Secrets such as API keys must be stored only in `.env`.

## Vector-store options

The application stores normalized embeddings and converts each store's squared L2 distance into the displayed similarity score `1 / (1 + distance)`. Higher scores mean closer chunks; this is a comparable retrieval score, not a model confidence value.

| Store            | Setup and persistence                                                                                     | Speed / best use case                                                                                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chroma (default) | No extra package beyond `requirements.txt`; persists in `chroma_db/` and keeps the existing project data. | Convenient document metadata and local development. Good default for this chatbot.                                                                                          |
| FAISS            | Set `VECTOR_DB=faiss`; persists its index and chunk metadata in `faiss_db/`.                              | Typically very fast local similarity search with a lightweight index. Best for a simple local, read-heavy knowledge base; metadata filtering is intentionally minimal here. |
