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
BACKEND_URL=http://127.0.0.1:8000
```

For OpenAI, change `LLM_PROVIDER` to `openai` and provide `OPENAI_API_KEY` in `.env`.

For the Streamlit frontend, `BACKEND_URL` specifies the FastAPI backend address.

For Docker Compose, `BACKEND_URL` is set automatically to:

```text
http://backend:8000
```

For cloud deployment, set `BACKEND_URL` to the public URL of the deployed FastAPI backend.

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
* `faiss_db/`

The `.env.example` file is committed as a safe configuration template.

Secrets such as API keys must be stored only in `.env` or as secret environment variables provided by the deployment platform.

## Environment variables

| Variable | Description | Example |
| --- | --- | --- |
| `LLM_PROVIDER` | Selects the LLM provider | `ollama` or `openai` |
| `OLLAMA_MODEL` | Local Ollama model name | `qwen2.5-coder:7b` |
| `OLLAMA_BASE_URL` | Ollama server address | `http://127.0.0.1:11434` |
| `OPENAI_API_KEY` | OpenAI API key when using OpenAI | Stored as a secret |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `VECTOR_DB` | Vector database implementation | `chroma` or `faiss` |
| `BACKEND_URL` | FastAPI backend address used by the frontend | `http://127.0.0.1:8000` |

## RAG pipeline

The chatbot follows a Retrieval-Augmented Generation pipeline:

```text
Documents
    ↓
Document loading
    ↓
Text cleaning
    ↓
Text chunking
    ↓
MiniLM embeddings
    ↓
Chroma / FAISS vector store
    ↓
Similarity retrieval
    ↓
Conversation-aware question rewriting
    ↓
Grounded prompt
    ↓
LLM
    ↓
Answer + source chunks
```

The system retrieves relevant document chunks before generating an answer. The prompt instructs the LLM to use the retrieved company knowledge and avoid inventing information that is not supported by the available documents.

## Document ingestion

Supported document formats:

* PDF
* TXT
* DOCX

Documents placed in the `data/` directory are processed into chunks with source metadata.

The current company knowledge base contains:

* `Employee_benefits.pdf`
* `Employee_handbook.txt`
* `Human_resource_policies.pdf`
* `Privacy_policy.docx`
* `User_guide.docx`

The application uses the local `all-MiniLM-L6-v2` sentence-transformer model to generate 384-dimensional embeddings.

## Retrieval

The application supports similarity-based document retrieval.

Retrieved results include:

* Source filename
* Page number when available
* Retrieved text
* Similarity score

The displayed similarity score is calculated from the vector-store distance:

```text
similarity = 1 / (1 + distance)
```

Higher values indicate that the retrieved chunk is closer to the query in embedding space.

The similarity score is a retrieval score and should **not** be interpreted as model confidence.

## Conversation-aware retrieval

The chatbot supports conversation history.

For follow-up questions, the recent conversation is used to create a standalone retrieval question before searching the knowledge base.

Example:

```text
User: What benefits are available to employees?

User: What about health insurance?

Retrieval question:
What health insurance benefits are available to employees?
```

This helps follow-up questions retain their conversational context while keeping retrieval focused on the relevant company documents.

## LLM providers

The application supports two LLM providers:

### Ollama

Ollama is the default provider for local development.

Example configuration:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### OpenAI

OpenAI can be selected through environment variables without changing application code.

Example:

```text
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=<your-secret-key>
```

The API key must be supplied through `.env` locally or through secret environment variables in a deployment platform.

## Vector-store options

The application supports ChromaDB and FAISS through a vector-store abstraction.

The application stores normalized embeddings and converts each store's squared L2 distance into the displayed similarity score `1 / (1 + distance)`.

| Store | Setup and persistence | Speed / best use case |
| --- | --- | --- |
| Chroma (default) | No extra package beyond `requirements.txt`; persists in `chroma_db/` and keeps the existing project data. | Convenient document metadata and local development. Good default for this chatbot. |
| FAISS | Set `VECTOR_DB=faiss`; persists its index and chunk metadata in `faiss_db/`. | Typically very fast local similarity search with a lightweight index. Best for a simple local, read-heavy knowledge base; metadata filtering is intentionally minimal here. |

## FastAPI backend

The backend provides the following endpoints:

### Health check

```text
GET /ping
```

Returns:

```json
{
  "status": "ok"
}
```

### Chat

```text
POST /chat
```

The endpoint accepts a question and optional conversation history.

The response includes:

* Generated answer
* Retrieved sources
* Retrieval question

### Document upload

```text
POST /upload
```

Supported uploads:

* PDF
* TXT
* DOCX

Uploaded documents are stored in the `data/` directory and indexed into the configured vector store.

## Input validation and error handling

The backend validates:

* Empty questions
* Maximum question length
* Conversation history size
* Individual history message length
* Supported upload file types

The API provides meaningful HTTP errors for common failures.

Examples include:

```text
400 Bad Request
503 Service Unavailable
500 Internal Server Error
429 Too Many Requests
```

LLM provider failures and vector-store failures are handled separately so that the API can return an appropriate service-unavailable response.

## Rate limiting

The `/chat` endpoint is protected with a rate limit of:

```text
10 requests per minute per client
```

This helps prevent excessive requests to the LLM and backend.

## CORS

The FastAPI backend includes CORS configuration for local Streamlit development.

The frontend communicates with the backend through the configurable:

```text
BACKEND_URL
```

This avoids hardcoding the backend address throughout the frontend application.

## Streamlit frontend

The Streamlit frontend provides:

* Chat interface
* Conversation history
* Document upload
* Retrieved source viewing
* Backend error handling
* Configurable backend URL

The frontend reads the backend address from:

```text
BACKEND_URL
```

Example local configuration:

```text
BACKEND_URL=http://127.0.0.1:8000
```

Docker Compose configuration:

```text
BACKEND_URL=http://backend:8000
```

Cloud deployment configuration should use the public FastAPI backend URL.

## Docker

The project includes:

* `Dockerfile.backend`
* `Dockerfile.frontend`
* `docker-compose.yml`

The Compose configuration contains separate backend and frontend services.

The backend exposes:

```text
8000
```

The frontend exposes:

```text
8501
```

The Compose configuration also provides persistent mounts for:

```text
data/
chroma_db/
faiss_db/
models/
```

This allows document data, vector-store data, and local embedding models to persist outside the containers.

The frontend communicates with the backend using:

```text
http://backend:8000
```

The backend can connect to a host Ollama server through:

```text
http://host.docker.internal:11434
```

### Start with Docker Compose

When Docker is available:

```powershell
docker compose up --build
```

Open the Streamlit frontend:

```text
http://localhost:8501
```

Stop the containers:

```powershell
docker compose down
```

Start them again without rebuilding:

```powershell
docker compose up
```

The persistent vector-store directories are mounted from the host so indexed data can survive container shutdown.

## Security and secret management

The project follows these security practices:

* Real `.env` files are excluded from Git.
* API keys are not stored in frontend source code.
* API keys are not printed in application logs.
* `.env.example` contains only placeholder values.
* Dockerfiles copy `.env.example`, not the real `.env`.
* Deployment API keys should be configured as secret environment variables.
* The repository should never contain real API keys or credentials.

## Testing

The project contains an automated RAG pipeline test suite.

Run:

```powershell
pytest -v
```

The current test suite contains 10 tests covering:

* Embedding generation
* Embedding normalization
* Context construction
* Empty retrieval results
* Document retrieval
* Retrieval result structure
* `top_k` behavior
* Empty-question handling
* RAG response structure

The current test run completed with:

```text
10 passed
```

## Adversarial testing

The project includes adversarial prompt tests to verify that the chatbot does not blindly follow instructions that conflict with the company knowledge-base grounding requirement.

The tests cover cases such as:

* Requests for information outside the knowledge base
* Attempts to override system behavior
* Requests for confidential information
* Requests for unsupported company information
* Prompt-injection-style instructions

The chatbot is expected to respond using supported company information and refuse or acknowledge when required information is not available.

## RAGAS evaluation

Task 16 uses a fixed 10-question evaluation dataset containing:

* Questions
* Ground-truth answers
* Expected contexts
* Retrieved contexts
* Generated answers

The evaluation measures:

* Context Precision
* Context Recall
* Answer Relevancy
* Faithfulness

The current evaluation produced valid Context Precision, Context Recall, and Answer Relevancy measurements for all 10 questions.

Faithfulness could not be completed because of a local evaluator-model compatibility issue involving the installed Transformers environment.

Therefore, Faithfulness is reported as unavailable rather than being treated as a score of zero.

The RAGAS evaluation files are stored under:

```text
evaluation/
```

## Query expansion

The project can improve retrieval for differently worded user questions by generating multiple search formulations before retrieval.

The query-expansion approach is:

```text
User question
      ↓
Generate 2–3 search rephrasings
      ↓
Retrieve documents for each query
      ↓
Merge results
      ↓
Remove duplicate chunks
      ↓
Use relevant chunks as context
      ↓
Generate grounded answer
```

This is intended to improve recall when the wording of a user question differs from the wording used in the company documents.

The original user question remains available for answer generation, while the expanded queries are used to improve document retrieval.

## Project evaluation observations

The evaluation showed that the system can retrieve relevant company-document material for the tested questions.

However, retrieval metrics such as Context Precision and Context Recall depend on:

* Query wording
* Chunk size
* Chunk overlap
* Embedding quality
* Number of retrieved chunks
* Expected-context annotation
* Similarity threshold

Therefore, RAGAS scores should be interpreted together with manual retrieval inspection rather than as a direct measure of overall chatbot accuracy.

## Known limitations

* Ollama is intended primarily for local development because a local Ollama server is not directly available to a cloud deployment.
* For production or cloud deployment, configure `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY` through the deployment platform's secret environment variables.
* The application depends on the quality of the retrieved document chunks. If relevant information is not retrieved, the generated answer may not contain the required information.
* Local LLM inference can be slow depending on available CPU, RAM, and model size.
* RAGAS Faithfulness evaluation is currently limited by a local evaluator-model compatibility issue in the installed Transformers environment.
* Docker deployment could not be fully runtime-verified on the development laptop because the Docker CLI/WSL environment was unavailable. The Dockerfiles and Compose configuration were created and checked structurally.
* A cloud deployment requires a publicly accessible backend URL and a separately configured frontend `BACKEND_URL`.
* Local vector databases such as ChromaDB and FAISS are intended for this project's development and small-scale knowledge-base use case rather than large distributed production workloads.

## Project structure

```text
RAG-Company-Knowledge-Chatbot/
│
├── backend/
│   ├── chat.py
│   ├── ingest.py
│   ├── llm_factory.py
│   ├── main.py
│   ├── prompts.py
│   ├── rag_chain.py
│   └── vector_store.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── Employee_benefits.pdf
│   ├── Employee_handbook.txt
│   ├── Human_resource_policies.pdf
│   ├── Privacy_policy.docx
│   └── User_guide.docx
│
├── evaluation/
│   ├── evaluate.py
│   ├── ragas_input.json
│   ├── ragas_results.json
│   └── test_dataset.json
│
├── models/
│   └── all-MiniLM-L6-v2/
│
├── tests/
│   ├── conftest.py
│   └── test_rag_pipeline.py
│
├── chroma_db/
├── faiss_db/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Overall workflow

```text
                 ┌──────────────────┐
                 │ Company Documents│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Load + Clean     │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Chunk Documents  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ MiniLM Embeddings│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Chroma / FAISS   │
                 └────────┬─────────┘
                          ↓
User → Streamlit → FastAPI → Query Expansion
                                  ↓
                            Vector Retrieval
                                  ↓
                            Context Building
                                  ↓
                              LLM Answer
                                  ↓
                         Answer + Sources
```

## Project status

The project currently implements a complete local RAG chatbot pipeline with:

* PDF, TXT, and DOCX ingestion
* Recursive text chunking
* Local MiniLM embeddings
* ChromaDB and FAISS vector-store support
* Similarity-based retrieval
* Conversation-aware retrieval
* Query expansion
* Grounded LLM answer generation
* Ollama and OpenAI provider support
* FastAPI backend
* Streamlit frontend
* Document upload and re-indexing
* Source attribution
* Input validation
* Error handling
* Rate limiting
* CORS configuration
* Automated tests
* Adversarial testing
* RAGAS evaluation
* Dockerfiles
* Docker Compose configuration
* Environment-based configuration
* Secret-management practices
