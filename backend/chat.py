from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend import rag_chain


# --------------------------------
# Create router
# --------------------------------

router = APIRouter()


# --------------------------------
# Request models
# --------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # User question must contain 1-1000 characters
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question, maximum 1000 characters."
    )

    # Prevent excessively large conversation history
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Previous conversation messages, maximum 20 messages."
    )


# --------------------------------
# Response models
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

    # --------------------------------
    # Check for empty or whitespace-only question
    # --------------------------------

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # --------------------------------
        # Call the RAG chain
        # --------------------------------

        result = rag_chain.ask_question(
            request.question.strip(),
            [message.model_dump() for message in request.history],
        )

        # --------------------------------
        # Validate RAG response
        # --------------------------------

        if not isinstance(result, dict):
            raise RuntimeError(
                "RAG chain returned an invalid response."
            )

        # --------------------------------
        # Get answer
        # --------------------------------

        final_answer = result.get("answer", "")

        if not final_answer:
            final_answer = (
                "I could not find an answer to your question."
            )

        # --------------------------------
        # Get sources
        # --------------------------------

        sources = result.get("sources", [])

        if not isinstance(sources, list):
            sources = []

        # --------------------------------
        # Return response
        # --------------------------------

        return ChatResponse(
            answer=final_answer,
            sources=sources
        )

    # --------------------------------
    # Re-raise intentional HTTP errors
    # --------------------------------

    except HTTPException:
        raise

    # --------------------------------
    # Handle unexpected errors
    # --------------------------------

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        ) from e

