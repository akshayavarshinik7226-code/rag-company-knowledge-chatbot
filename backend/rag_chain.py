"""Retrieval-augmented chat pipeline for the company knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from backend.llm_factory import get_llm
from backend.vector_store import get_vector_store


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"
MAX_HISTORY_MESSAGES = 6  # The three most recent turns only.

print("Loading embedding model...")
embedding_model = SentenceTransformer(str(MODEL_PATH))
print("Embedding model loaded successfully.")


def create_embedding(text: str) -> list[float]:
    """Convert text to a normalized 384-dimensional embedding."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


def _prepare_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize documents into the format used by either vector-store backend."""
    prepared = []
    for index, document in enumerate(documents):
        text = document.get("text", "").strip()
        if not text:
            continue
        source = document.get("source", "Unknown")
        page = document.get("page")
        prepared.append(
            {
                "id": f"{source}_{page}_{index}",
                "text": text,
                "metadata": {"source": source, "page": page},
            }
        )
    return prepared


def add_documents(documents: list[dict[str, Any]]) -> int:
    """Embed and persist chunks in the vector store selected by VECTOR_DB."""
    prepared = _prepare_documents(documents)
    if not prepared:
        return 0
    embeddings = [create_embedding(document["text"]) for document in prepared]
    return get_vector_store().add(prepared, embeddings)


def search_documents(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve chunks, each with a 0..1 distance-derived similarity score."""
    if not question.strip():
        return []
    return get_vector_store().search(create_embedding(question), top_k)


def build_context(results: list[dict[str, Any]]) -> str:
    """Combine retrieved chunks into grounded model context."""
    context_parts = []
    for result in results:
        citation = result["source"]
        if result.get("page"):
            citation += f", page {result['page']}"
        context_parts.append(f"[Source: {citation}]\n{result['text']}")
    return "\n\n".join(context_parts)


def _recent_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Accept only recent, well-formed turns so unrelated old chat is excluded."""
    if not history:
        return []
    cleaned = [
        {"role": item["role"], "content": item["content"].strip()}
        for item in history
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]
    return cleaned[-MAX_HISTORY_MESSAGES:]


def _history_text(history: list[dict[str, str]]) -> str:
    return "\n".join(f"{item['role'].title()}: {item['content']}" for item in history)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content.strip() if isinstance(content, str) else str(content).strip()


def resolve_retrieval_question(question: str, history: list[dict[str, str]]) -> str:
    """Use recent context only when needed to make a follow-up searchable."""
    if not history:
        return question
    prompt = (
        "Rewrite the latest user question as a standalone search query only when "
        "it depends on the recent conversation (for example, uses 'it', 'that', "
        "or 'what about'). Otherwise return it unchanged. Do not answer it and do "
        "not add facts.\n\nRecent conversation:\n"
        f"{_history_text(history)}\n\nLatest question: {question}"
    )
    standalone = _message_text(get_llm().invoke(prompt))
    return standalone or question


def generate_answer(question: str, results: list[dict[str, Any]], history: list[dict[str, str]]) -> str:
    if not results:
        return "I couldn't find relevant information in the company knowledge base."
    prompt = (
        "You are a company knowledge assistant. Answer only from the retrieved "
        "company-document context. If the context does not answer the question, say "
        "so clearly. Be concise and do not invent policy details.\n\n"
        f"Retrieved context:\n{build_context(results)}\n\n"
        f"Recent conversation (for conversational phrasing only):\n{_history_text(history) or '(none)'}\n\n"
        f"Question: {question}"
    )
    answer = _message_text(get_llm().invoke(prompt))
    return answer or "I could not generate an answer from the retrieved documents."


def ask_question(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Retrieve company knowledge and answer using the configured LLM provider."""
    question = question.strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": [], "retrieval_question": ""}
    recent_history = _recent_history(history)
    retrieval_question = resolve_retrieval_question(question, recent_history)
    results = search_documents(retrieval_question, top_k=5)
    return {
        "answer": generate_answer(question, results, recent_history),
        "sources": results,
        "retrieval_question": retrieval_question,
    }


if __name__ == "__main__":
    print("RAG chain loaded successfully.")
    print("Documents currently stored:", get_vector_store().count())
