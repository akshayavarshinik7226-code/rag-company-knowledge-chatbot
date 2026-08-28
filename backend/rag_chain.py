```python
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Project paths
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="company_documents"
)


# OpenAI client
llm = OpenAI()


def search(query, k=3):
    """Retrieve the most relevant document chunks."""

    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


def answer(question):
    """Generate an answer using retrieved document context."""

    documents, metadatas = search(question, k=3)

    context = "\n\n".join(documents)

    prompt = f"""
Answer the question using ONLY the context below.

If the answer is not present in the context, say:
"I don't know based on the provided company documents."

Context:
{context}

Question:
{question}
"""

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a company knowledge assistant. Answer only from the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": metadatas
    }


if __name__ == "__main__":
    question = "What is the annual leave policy?"

    result = answer(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(source)
```
