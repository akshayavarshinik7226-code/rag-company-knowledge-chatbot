import pytest

from backend.rag_chain import (
    build_context,
    create_embedding,
    search_documents,
    ask_question,
)


def test_create_embedding_returns_vector():
    embedding = create_embedding("What are the employee benefits?")

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)


def test_create_embedding_is_normalized():
    embedding = create_embedding(
        "Employee benefits include health insurance."
    )

    magnitude = sum(value * value for value in embedding) ** 0.5

    assert magnitude == pytest.approx(1.0, abs=0.01)


def test_build_context_with_one_result():
    results = [
        {
            "source": "Employee_benefits.pdf",
            "page": 1,
            "text": "Employees receive health insurance.",
            "similarity_score": 0.8,
        }
    ]

    context = build_context(results)

    assert "Employee_benefits.pdf" in context
    assert "page 1" in context
    assert "Employees receive health insurance." in context


def test_build_context_with_multiple_results():
    results = [
        {
            "source": "Employee_benefits.pdf",
            "page": 1,
            "text": "Health insurance is provided.",
            "similarity_score": 0.8,
        },
        {
            "source": "Employee_handbook.txt",
            "page": None,
            "text": "Employees receive paid leave.",
            "similarity_score": 0.7,
        },
    ]

    context = build_context(results)

    assert "Health insurance is provided." in context
    assert "Employees receive paid leave." in context
    assert "Employee_benefits.pdf" in context
    assert "Employee_handbook.txt" in context


def test_build_context_empty_results():
    context = build_context([])

    assert context == ""


def test_search_documents_returns_list():
    results = search_documents(
        "What employee benefits are available?",
        top_k=5,
    )

    assert isinstance(results, list)
    assert len(results) <= 5


def test_search_documents_contains_source_information():
    results = search_documents(
        "What employee benefits are available?",
        top_k=5,
    )

    assert len(results) > 0

    first_result = results[0]

    assert "source" in first_result
    assert "text" in first_result
    assert "similarity_score" in first_result


def test_search_documents_respects_top_k():
    results = search_documents(
        "What is the company leave policy?",
        top_k=3,
    )

    assert len(results) <= 3


def test_ask_question_empty_input():
    result = ask_question("")

    assert isinstance(result, dict)
    assert result["answer"] == "Please enter a question."
    assert result["sources"] == []


def test_ask_question_returns_expected_structure():
    result = ask_question(
        "What benefits are available to employees?"
    )

    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert "retrieval_question" in result

    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["retrieval_question"], str)