```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
```
