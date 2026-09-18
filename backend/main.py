from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


from backend.chat import router as chat_router, limiter
from backend.ingest import ingest_file


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="Company Knowledge Chatbot API",
    description="RAG-based chatbot for company documents",
    version="1.0",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        DATA_DIR.mkdir(exist_ok=True)

        filename = Path(file.filename or "").name

        if not filename or Path(filename).suffix.lower() not in {
            ".pdf",
            ".txt",
            ".docx",
        }:
            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT, and DOCX files are supported.",
            )

        file_path = DATA_DIR / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        indexed_chunks = ingest_file(file_path)

        if not indexed_chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the uploaded file.",
            )

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": filename,
            "indexed_chunks": indexed_chunks,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Upload and indexing failed. Please try again later.",
        ) from error


app.include_router(chat_router)



app.include_router(chat_router)

