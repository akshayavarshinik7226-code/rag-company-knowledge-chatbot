from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from backend.chat import router as chat_router
from backend.ingest import ingest_file


app = FastAPI(
    title="Company Knowledge Chatbot API",
    description="RAG-based chatbot for company documents",
    version="1.0"
)


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        DATA_DIR = BASE_DIR / "data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        filename = Path(file.filename or "").name
        if not filename or Path(filename).suffix.lower() not in {".pdf", ".txt", ".docx"}:
            raise HTTPException(status_code=400, detail="Only PDF, TXT, and DOCX files are supported.")

        file_path = DATA_DIR / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        indexed_chunks = ingest_file(file_path)
        if not indexed_chunks:
            raise HTTPException(status_code=400, detail="No readable text was found in the uploaded document.")

        return {
            "message": f"{filename} uploaded and indexed successfully.",
            "filename": filename,
            "indexed_chunks": indexed_chunks,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload and indexing failed: {str(e)}") from e


app.include_router(chat_router)
