"""Streamlit interface for the Company Knowledge Chatbot."""

from pathlib import Path
import sys

# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

import streamlit as st

from backend.ingest import ingest_file
from backend.vector_store import get_vector_store

from backend.rag_chain import (
    LLMServiceError,
    VectorStoreServiceError,
    ask_question,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Company Knowledge Chatbot",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# Automatically build knowledge base
# ============================================================

try:
    vector_store = get_vector_store()

    # Only ingest documents when the vector database is empty.
    if vector_store.count() == 0:
        data_dir = PROJECT_ROOT / "data"

        if data_dir.exists():
            for file_path in data_dir.iterdir():
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in {".pdf", ".txt", ".docx"}
                ):
                    ingest_file(file_path)

except Exception as error:
    print("Automatic knowledge-base initialization error:", repr(error))


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Source display
# ============================================================

def render_sources(sources):
    if not sources:
        return

    with st.expander("View sources"):
        st.caption(
            "Similarity is a distance-derived score from 0 to 1; "
            "higher is closer."
        )

        for source in sources:
            filename = source.get("source", "Unknown")
            page = source.get("page")
            score = source.get("similarity_score")

            label = f"**{filename}**"

            if page:
                label += f" — page {page}"

            if score is not None:
                label += f" — similarity {score:.4f}"

            st.markdown(label)

            st.code(
                source.get("text", ""),
                language=None,
            )


# ============================================================
# Main interface
# ============================================================

st.title("🤖 Company Knowledge Chatbot")

st.caption(
    "Ask questions about company policies and documents."
)


# ============================================================
# Display previous messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(
                message.get("sources", [])
            )


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    "Ask a question about the company documents..."
)


if question:

    question = question.strip()

    if not question:

        st.warning("Please enter a question.")

    else:

        history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in st.session_state.messages[-6:]
            if message["role"] in {"user", "assistant"}
        ]

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching company documents..."
                ):

                    result = ask_question(
                        question=question,
                        history=history,
                    )

                answer = result.get(
                    "answer",
                    "No answer was returned.",
                )

                sources = result.get(
                    "sources",
                    [],
                )

                st.markdown(answer)

                render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except LLMServiceError:

                error = (
                    "The configured LLM provider is currently "
                    "unavailable. Please try again later."
                )

                st.error(error)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error,
                        "sources": [],
                    }
                )

            except VectorStoreServiceError:

                error = (
                    "The company knowledge base is currently "
                    "unavailable. Please try again later."
                )

                st.error(error)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error,
                        "sources": [],
                    }
                )

            except Exception as error:

                error_message = (
                    "An unexpected error occurred while "
                    "processing your question."
                )

                st.error(error_message)

                print(
                    "Unexpected Streamlit error:",
                    repr(error),
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# ============================================================
# Sidebar - document upload
# ============================================================

st.sidebar.header("📄 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, TXT, or DOCX",
    type=["pdf", "txt", "docx"],
)


if uploaded_file is not None:

    if st.sidebar.button("Upload and Index"):

        try:

            with st.sidebar.spinner(
                "Uploading and indexing..."
            ):

                data_dir = PROJECT_ROOT / "data"

                data_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                safe_filename = Path(
                    uploaded_file.name
                ).name

                file_path = data_dir / safe_filename

                file_path.write_bytes(
                    uploaded_file.getvalue()
                )

                indexed_chunks = ingest_file(
                    file_path
                )

            if not indexed_chunks:

                st.sidebar.error(
                    "No readable text was found in the uploaded file."
                )

            else:

                st.sidebar.success(
                    "Document uploaded and indexed successfully."
                )

                st.sidebar.info(
                    f"Indexed chunks: {indexed_chunks}"
                )

        except VectorStoreServiceError:

            st.sidebar.error(
                "The company knowledge base is currently unavailable."
            )

        except Exception as error:

            st.sidebar.error(
                "Upload/indexing failed. Please try again."
            )

            print(
                "Upload/indexing error:",
                repr(error),
            )


# ============================================================
# Sidebar footer
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Company Knowledge Chatbot"
)

st.sidebar.caption(
    "RAG-based document question answering"
)