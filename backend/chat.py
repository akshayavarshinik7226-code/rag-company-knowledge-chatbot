"""FastAPI chat endpoint for the company knowledge chatbot."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.rag_chain import (
    LLMServiceError,
    VectorStoreServiceError,
    ask_question,
)


# Router for chat-related endpoints
router = APIRouter()

# Rate limiter
# Each client IP is limited to 10 chat requests per minute.
limiter = Limiter(key_func=get_remote_address)


class ChatMessage(BaseModel):
    """A single message from the conversation history."""

    role: Literal["user", "assistant"]

    content: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class ChatRequest(BaseModel):
    """Request body accepted by the /chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
    )


class ChatResponse(BaseModel):
    """Response returned by the /chat endpoint."""

    answer: str
    sources: list[dict]
    retrieval_question: str = ""


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(
    request: Request,
    chat_request: ChatRequest,
) -> ChatResponse:
    """
    Process a user question using the RAG pipeline.

    Validation, rate limiting, and safe API errors
    are handled at the endpoint level.
    """

    question = chat_request.question.strip()

    # Reject whitespace-only questions.
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # Convert Pydantic history objects into the format
    # expected by the RAG pipeline.
    history = [
        {
            "role": message.role,
            "content": message.content.strip(),
        }
        for message in chat_request.history
        if message.content.strip()
    ]

    try:
        result = ask_question(
            question=question,
            history=history,
        )

        return ChatResponse(
            answer=result.get(
                "answer",
                "I couldn't find that information in the company knowledge base.",
            ),
            sources=result.get("sources", []),
            retrieval_question=result.get(
                "retrieval_question",
                question,
            ),
        )

    except LLMServiceError as error:
        raise HTTPException(
            status_code=503,
            detail="The configured LLM provider is currently unavailable.",
        ) from error

    except VectorStoreServiceError as error:
        raise HTTPException(
            status_code=503,
            detail="The company knowledge base is currently unavailable.",
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the request.",
        ) from error
