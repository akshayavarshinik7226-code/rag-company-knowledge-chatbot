"""Streamlit interface for the Company Knowledge Chatbot."""

import os

import requests
import streamlit as st


st.set_page_config(page_title="Company Knowledge Chatbot", page_icon="🤖", layout="centered")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT_TIMEOUT_SECONDS = 90
UPLOAD_TIMEOUT_SECONDS = 180

st.title("🤖 Company Knowledge Chatbot")
st.caption("Ask questions about company policies and documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    """Show clean source names and the exact chunks returned by /chat."""
    if not sources:
        return
    with st.expander(" View sources"):
        st.caption("Similarity is a distance-derived score from 0 to 1; higher is closer.")
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
    try:
        return response.json().get("detail") or response.json().get("message") or fallback
    except (ValueError, requests.RequestException):
        return fallback


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


if question := st.chat_input("Ask a question about the company documents..."):
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-6:]
        if message["role"] in {"user", "assistant"}
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching company documents..."):
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": question, "history": history},
                    timeout=CHAT_TIMEOUT_SECONDS,
                )
            if not response.ok:
                raise RuntimeError(backend_detail(response, "Backend returned an error."))
            payload = response.json()
            answer = payload.get("answer", "No answer was returned.")
            sources = payload.get("sources", [])
            st.markdown(answer)
            render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except requests.exceptions.Timeout:
            error = "⏳ The backend took too long to respond. Please try again."
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "sources": []})
        except requests.exceptions.ConnectionError:
            error = "🔌 Could not connect to FastAPI. Make sure the backend is running."
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "sources": []})
        except (RuntimeError, ValueError) as error:
            message = f"❌ {error}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})
        except requests.RequestException as error:
            message = f"❌ Request failed: {error}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})


st.sidebar.title("📄 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Choose a document", type=["pdf", "txt", "docx"])
if uploaded_file and st.sidebar.button("Upload & Index", type="primary"):
    try:
        with st.sidebar, st.spinner("Uploading and indexing document..."):
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        if not response.ok:
            st.sidebar.error(f"❌ {backend_detail(response, 'Backend rejected the upload.')}")
        else:
            payload = response.json()
            st.sidebar.success(payload.get("message", "Document uploaded and indexed successfully."))
    except requests.exceptions.Timeout:
        st.sidebar.error("⏳ Upload timed out. Please check the backend.")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("🔌 Could not connect to FastAPI. Make sure the backend is running.")
    except requests.RequestException as error:
        st.sidebar.error(f"❌ Upload failed: {error}")
