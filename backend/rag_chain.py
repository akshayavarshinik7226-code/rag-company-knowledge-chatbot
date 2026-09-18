"""Retrieval-augmented chat pipeline for the company knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from backend.llm_factory import get_llm
from backend.vector_store import get_vector_store
from backend.prompts import build_grounded_prompt


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"

# Keep only the three most recent conversation turns.
MAX_HISTORY_MESSAGES = 6

# Maximum number of alternative search queries generated.
MAX_EXPANDED_QUERIES = 3

# Number of chunks retrieved for each search query.
TOP_K_PER_QUERY = 5

# Maximum number of final unique chunks sent to the answer-generation LLM.
FINAL_TOP_K = 5


class LLMServiceError(RuntimeError):
    """Raised when the configured LLM provider cannot process a request."""


class VectorStoreServiceError(RuntimeError):
    """Raised when the configured vector store cannot process a request."""


# The model is NOT loaded when the backend starts.
# It will only be loaded when create_embedding() is actually called.
embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model only when it is actually needed."""

    global embedding_model

    if embedding_model is None:
        print("Loading embedding model...")

        embedding_model = SentenceTransformer(
            str(MODEL_PATH)
        )

        print("Embedding model loaded successfully.")

    return embedding_model


def create_embedding(text: str) -> list[float]:
    """Convert text to a normalized 384-dimensional embedding."""

    model = get_embedding_model()

    return model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


def _prepare_documents(
    documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
                "metadata": {
                    "source": source,
                    "page": page
                },
            }
        )

    return prepared


def add_documents(
    documents: list[dict[str, Any]]
) -> int:
    """Embed and persist chunks in the configured vector store."""

    prepared = _prepare_documents(documents)

    if not prepared:
        return 0

    try:
        embeddings = [
            create_embedding(document["text"])
            for document in prepared
        ]

        return get_vector_store().add(
            prepared,
            embeddings
        )

    except Exception as error:
        raise VectorStoreServiceError(
            "The vector store is currently unavailable."
        ) from error


def search_documents(
    question: str,
    top_k: int = TOP_K_PER_QUERY
) -> list[dict[str, Any]]:
    """Retrieve chunks from the configured vector store."""

    if not question.strip():
        return []

    try:
        embedding = create_embedding(question)

        return get_vector_store().search(
            embedding,
            top_k
        )

    except Exception as error:
        raise VectorStoreServiceError(
            "The vector store is currently unavailable."
        ) from error


def build_context(
    results: list[dict[str, Any]]
) -> str:
    """Combine retrieved chunks into grounded model context."""

    context_parts = []

    for result in results:
        citation = result["source"]

        if result.get("page"):
            citation += f", page {result['page']}"

        context_parts.append(
            f"[Source: {citation}]\n{result['text']}"
        )

    return "\n\n".join(context_parts)


def _recent_history(
    history: list[dict[str, str]] | None
) -> list[dict[str, str]]:
    """Accept only recent, well-formed turns."""

    if not history:
        return []

    cleaned = [
        {
            "role": item["role"],
            "content": item["content"].strip()
        }
        for item in history
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]

    return cleaned[-MAX_HISTORY_MESSAGES:]


def _history_text(
    history: list[dict[str, str]]
) -> str:
    """Convert conversation history into readable text."""

    return "\n".join(
        f"{item['role'].title()}: {item['content']}"
        for item in history
    )


def _message_text(message: Any) -> str:
    """Extract text from a LangChain message or regular value."""

    content = getattr(message, "content", message)

    return (
        content.strip()
        if isinstance(content, str)
        else str(content).strip()
    )


def resolve_retrieval_question(
    question: str,
    history: list[dict[str, str]]
) -> str:
    """
    Use recent conversation context when needed to make a follow-up
    question standalone and searchable.
    """

    if not history:
        return question

    prompt = (
        "Rewrite the latest user question as a standalone search query only "
        "when it depends on the recent conversation. For example, rewrite "
        "questions containing words such as 'it', 'that', 'this', or "
        "'what about' when necessary.\n"
        "Otherwise return the question unchanged.\n"
        "Do not answer the question.\n"
        "Do not add facts.\n"
        "Return only the standalone search query.\n\n"
        "Recent conversation:\n"
        f"{_history_text(history)}\n\n"
        f"Latest question: {question}"
    )

    try:
        standalone = _message_text(
            get_llm().invoke(prompt)
        )

    except Exception as error:
        raise LLMServiceError(
            "The configured LLM provider is currently unavailable."
        ) from error

    return standalone or question


def expand_search_queries(
    question: str,
    max_queries: int = MAX_EXPANDED_QUERIES
) -> list[str]:
    """
    Generate alternative search queries for the same user question.

    The original question is always retained.
    Generated queries are only used for document retrieval.
    """

    if not question.strip():
        return []

    prompt = (
        "Generate up to 2 alternative search queries for the user's "
        "question.\n"
        "These queries will be used only to search a company knowledge base.\n\n"
        "Rules:\n"
        "1. Keep the exact meaning of the original question.\n"
        "2. Do not answer the question.\n"
        "3. Do not add facts or assumptions.\n"
        "4. Use different wording where useful.\n"
        "5. Return one query per line.\n"
        "6. Return only the search queries.\n\n"
        f"User question: {question}"
    )

    try:
        response = _message_text(
            get_llm().invoke(prompt)
        )

    except Exception as error:
        raise LLMServiceError(
            "The configured LLM provider is currently unavailable."
        ) from error

    queries = []

    for line in response.splitlines():
        cleaned = line.strip()

        cleaned = cleaned.lstrip("-•* ")

        if cleaned:
            parts = cleaned.split(maxsplit=1)

            if (
                len(parts) == 2
                and parts[0][:-1].isdigit()
                and parts[0][-1:] in {".", ")"}
            ):
                cleaned = parts[1].strip()

        if cleaned:
            queries.append(cleaned)

    unique_queries = []

    # Always search using the original query first.
    candidates = [question, *queries]

    for query in candidates:
        normalized = query.strip()

        if not normalized:
            continue

        if any(
            normalized.lower() == existing.lower()
            for existing in unique_queries
        ):
            continue

        unique_queries.append(normalized)

        if len(unique_queries) >= max_queries:
            break

    return unique_queries


def _merge_and_deduplicate_results(
    results: list[dict[str, Any]],
    max_results: int = FINAL_TOP_K
) -> list[dict[str, Any]]:
    """
    Merge retrieval results from multiple queries and remove duplicate chunks.
    """

    unique_results = []
    seen_chunks = set()

    for result in results:
        text = result.get("text", "").strip()

        chunk_key = (
            result.get("source"),
            result.get("page"),
            text
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)
        unique_results.append(result)

    unique_results.sort(
        key=lambda item: item.get("similarity_score", 0),
        reverse=True
    )

    return unique_results[:max_results]


def retrieve_with_query_expansion(
    question: str,
    top_k_per_query: int = TOP_K_PER_QUERY,
    final_top_k: int = FINAL_TOP_K
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Expand the retrieval query, search each query, merge the results,
    remove duplicates, and keep the strongest chunks.
    """

    search_queries = expand_search_queries(
        question,
        max_queries=MAX_EXPANDED_QUERIES
    )

    if not search_queries:
        return [], []

    all_results = []

    for search_query in search_queries:
        results = search_documents(
            search_query,
            top_k=top_k_per_query
        )

        all_results.extend(results)

    merged_results = _merge_and_deduplicate_results(
        all_results,
        max_results=final_top_k
    )

    return merged_results, search_queries


def generate_answer(
    question: str,
    results: list[dict[str, Any]],
    history: list[dict[str, str]]
) -> str:
    """Generate a grounded answer using retrieved company knowledge."""

    if not results:
        return (
            "I couldn't find that information in the company "
            "knowledge base."
        )

    context = build_context(results)

    history_text = (
        _history_text(history)
        or "(none)"
    )

    prompt = build_grounded_prompt(
        question=question,
        context=context,
        history=history_text,
    )

    try:
        answer = _message_text(
            get_llm().invoke(prompt)
        )

    except Exception as error:
        raise LLMServiceError(
            "The configured LLM provider is currently unavailable."
        ) from error

    return (
        answer
        or "I couldn't find that information in the company knowledge base."
    )


def ask_question(
    question: str,
    history: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """
    Retrieve company knowledge using query expansion and generate
    a grounded answer.
    """

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieval_question": ""
        }

    recent_history = _recent_history(history)

    retrieval_question = resolve_retrieval_question(
        question,
        recent_history
    )

    results, search_queries = retrieve_with_query_expansion(
        retrieval_question,
        top_k_per_query=TOP_K_PER_QUERY,
        final_top_k=FINAL_TOP_K
    )

    return {
        "answer": generate_answer(
            question,
            results,
            recent_history
        ),
        "sources": results,
        "retrieval_question": retrieval_question,
        "search_queries": search_queries,
    }


if __name__ == "__main__":
    print("RAG chain loaded successfully.")

    print(
        "Documents currently stored:",
        get_vector_store().count()
    )