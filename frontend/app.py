import streamlit as st
import requests


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Company Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Backend configuration
# -----------------------------
BACKEND_URL = "http://127.0.0.1:8000"


# -----------------------------
# Page title
# -----------------------------
st.title("🤖 Company Knowledge Chatbot")
st.caption("Ask questions about company policies and documents.")


# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                st.caption("📚 Sources:")

                for source in sources:
                    st.caption(f"• {source}")


# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input(
    "Ask a question about the company documents..."
)


# -----------------------------
# Process question
# -----------------------------
if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching company documents..."):

            try:

                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "question": question
                    },
                    timeout=60
                )

                # -----------------------------
                # Backend error
                # -----------------------------
                if response.status_code != 200:

                    try:
                        error_detail = response.json().get(
                            "detail",
                            "Backend returned an error."
                        )

                    except Exception:
                        error_detail = "Backend returned an error."

                    error_message = f"❌ {error_detail}"

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        }
                    )

                # -----------------------------
                # Successful response
                # -----------------------------
                else:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer was returned."
                    )

                    sources = data.get(
                        "sources",
                        []
                    )

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if sources:

                        st.caption("📚 Sources:")

                        for source in sources:
                            st.caption(f"• {source}")

                    # Save assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

            # -----------------------------
            # Timeout
            # -----------------------------
            except requests.exceptions.Timeout:

                error_message = (
                    "⏳ The backend took too long to respond. "
                    "Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )

            # -----------------------------
            # Backend unavailable
            # -----------------------------
            except requests.exceptions.ConnectionError:

                error_message = (
                    "🔌 Could not connect to the FastAPI backend. "
                    "Make sure the backend is running."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )

            # -----------------------------
            # Unexpected error
            # -----------------------------
            except Exception as e:

                error_message = (
                    f"❌ Unexpected error: {str(e)}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📄 Upload Documents")

uploaded_file = st.sidebar.file_uploader(
    "Choose a document",
    type=["pdf", "txt", "docx"]
)


# -----------------------------
# Upload and index document
# -----------------------------
if uploaded_file is not None:

    if st.sidebar.button("Upload & Index"):

        try:

            with st.spinner(
                "Uploading and indexing document..."
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=files,
                    timeout=120
                )

                # Check response
                if response.status_code != 200:

                    try:
                        detail = response.json().get(
                            "detail",
                            "Backend rejected the upload."
                        )

                    except Exception:
                        detail = "Backend rejected the upload."

                    st.sidebar.error(
                        f"❌ {detail}"
                    )

                else:

                    data = response.json()

                    st.sidebar.success(
                        data.get(
                            "message",
                            "Document uploaded and indexed successfully!"
                        )
                    )

        except requests.exceptions.Timeout:

            st.sidebar.error(
                "⏳ Upload timed out. Please check the backend."
            )

        except requests.exceptions.ConnectionError:

            st.sidebar.error(
                "🔌 Could not connect to FastAPI. "
                "Make sure the backend is running."
            )

        except Exception as e:

            st.sidebar.error(
                f"❌ Upload failed: {str(e)}"
            )