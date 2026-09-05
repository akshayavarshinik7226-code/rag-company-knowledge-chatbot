from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from chat import router as chat_router


# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(
    title="Company Knowledge Chatbot API",
    description="Backend API for the RAG Company Knowledge Chatbot",
    version="1.0.0"
)


# -----------------------------
# Include chat router
# -----------------------------
app.include_router(chat_router)


# -----------------------------
# Upload directory
# -----------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# -----------------------------
# Health check
# -----------------------------
@app.get("/ping")
def ping():
    return {"status": "ok"}


# -----------------------------
# Document upload endpoint
# -----------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # Check file type
    allowed_extensions = {".pdf", ".txt", ".docx"}

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and DOCX files are supported."
        )

    # Create safe file path
    file_path = UPLOAD_DIR / file.filename

    try:

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -----------------------------------------
        # Re-ingest uploaded document
        # -----------------------------------------
        #
        # IMPORTANT:
        # Replace the next section with the
        # ingestion function from your Task 7 code.
        #
        # Example:
        #
        # from ingestion import ingest_document
        # ingest_document(str(file_path))
        #
        # -----------------------------------------

        return {
            "message": f"{file.filename} uploaded successfully.",
            "filename": file.filename
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
