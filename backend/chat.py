from fastapi import APIRouter, HTTPException
from pydantic import BaseModel# -----------------------------
# Upload document endpoint
# -----------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    try:

        # Location of the data folder
        BASE_DIR = Path(__file__).resolve().parent.parent
        DATA_DIR = BASE_DIR / "data"

        # Create data folder if it does not exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = DATA_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "message": f"{file.filename} uploaded successfully.",
            "filename": file.filename
        }

    except Exception as e:

        return {
            "message": f"Upload failed: {str(e)}"
        }
from rag_chain import answer


router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Reject empty questions
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        result = answer(request.question)

        # Convert source metadata to strings
        sources = [
            str(source)
            for source in result.get("sources", [])
        ]

        return ChatResponse(
            answer=result["answer"],
            sources=sources
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

