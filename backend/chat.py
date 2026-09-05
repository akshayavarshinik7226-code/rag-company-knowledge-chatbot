from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend import rag_chain


# --------------------------------
# Create router
# --------------------------------

router = APIRouter()


# --------------------------------
# Request model
# --------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


# --------------------------------
# Response model
# --------------------------------

class SourceChunk(BaseModel):
    source: str
    page: str | None = None
    text: str
    similarity_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# --------------------------------
# Chat endpoint
# --------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # Check for empty question
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # Call the RAG chain
        result = rag_chain.ask_question(
            request.question,
            [message.model_dump() for message in request.history],
        )

        # Make sure we received a dictionary
        if not isinstance(result, dict):
            raise Exception("RAG chain returned an invalid response.")

        # Get answer
        final_answer = result.get("answer", "")

        # Get sources
        sources = result.get("sources", [])

        # Check answer
        if not final_answer:
            final_answer = "I could not find an answer to your question."

        return ChatResponse(
            answer=final_answer,
            sources=sources
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        ) from e

