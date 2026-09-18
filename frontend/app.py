"""Streamlit interface for the Company Knowledge Chatbot."""

import os

import requests
import streamlit as st


st.set_page_config(
    page_title="Company Knowledge Chatbot",
    page_icon="🤖",
    layout="centered",
)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

CHAT_TIMEOUT_SECONDS = 240
UPLOAD_TIMEOUT_SECONDS = 180

st.title("🤖 Company Knowledge Chatbot")
st.caption("Ask questions about company policies and documents.")


if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    """Display retrieved source chunks."""
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
            st.code(source.get("text", ""), language=None)


def backend_detail(response, fallback):
    """Get an error message returned by FastAPI."""
    try:
        data = response.json()
        return data.get("detail") or data.get("message") or fallback
    except (ValueError, requests.RequestException):
        return fallback


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


# Chat section.
question = st.chat_input(
    "Ask a question about the company documents..."
)

if question:

    history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages[-6:]
        if message["role"] in {"user", "assistant"}
    ]

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching company documents..."):

                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "question": question,
                        "history": history,
                    },
                    timeout=CHAT_TIMEOUT_SECONDS,
                )

            if not response.ok:
                raise RuntimeError(
                    backend_detail(
                        response,
                        "Backend returned an error.",
                    )
                )

            payload = response.json()

            answer = payload.get(
                "answer",
                "No answer was returned.",
            )

            sources = payload.get(
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

        except requests.exceptions.Timeout:

            error = (
                "The backend took too long to respond. "
                "Please try again."
            )

            st.error(error)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error,
                    "sources": [],
                }
            )

        except requests.exceptions.ConnectionError:

            error = (
                "Could not connect to FastAPI. "
                "Make sure the backend is running."
            )

            st.error(error)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error,
                    "sources": [],
                }
            )

        except RuntimeError as error:

            st.error(str(error))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": str(error),
                    "sources": [],
                }
            )

        except ValueError:

            error = "The backend returned an invalid response."

            st.error(error)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error,
                    "sources": [],
                }
            )

        except requests.RequestException:

            error = (
                "A network error occurred while contacting "
                "the backend."
            )

            st.error(error)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error,
                    "sources": [],
                }
            )


# Document upload section.
st.sidebar.header("Document Upload")

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

                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    timeout=UPLOAD_TIMEOUT_SECONDS,
                )

            if not response.ok:
                raise RuntimeError(
                    backend_detail(
                        response,
                        "Upload failed.",
                    )
                )

            payload = response.json()

            st.sidebar.success(
                payload.get(
                    "message",
                    "Document uploaded successfully.",
                )
            )

            indexed_chunks = payload.get(
                "indexed_chunks"
            )

            if indexed_chunks is not None:
                st.sidebar.info(
                    f"Indexed chunks: {indexed_chunks}"
                )

        except requests.exceptions.Timeout:

            st.sidebar.error(
                "Upload timed out. Please try again."
            )

        except requests.exceptions.ConnectionError:

            st.sidebar.error(
                "Could not connect to FastAPI. "
                "Make sure the backend is running."
            )

        except RuntimeError as error:

            st.sidebar.error(str(error))

        except requests.RequestException:

            st.sidebar.error(
                "A network error occurred during upload."
            )