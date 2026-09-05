from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag_chain import add_documents


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 2. Load documents
# ============================================================

def load_document(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif suffix == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))

    else:
        return []

    return loader.load()


# ============================================================
# 3. Split documents
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


def create_chunks_for_file(file_path: Path):
    """Load one supported document and return the existing 500-character chunks."""
    chunks = []
    for document in load_document(file_path):
        page = document.metadata.get("page")
        for text in text_splitter.split_text(document.page_content):
            if text.strip():
                chunks.append(
                    {
                        "text": text,
                        "source": file_path.name,
                        "page": page + 1 if page is not None else None,
                    }
                )
    return chunks


def ingest_file(file_path: Path):
    """Index a single file; used by both the command line and /upload."""
    chunks = create_chunks_for_file(file_path)
    if not chunks:
        return 0
    return add_documents(chunks)


# ============================================================
# 4. Main ingestion process
# ============================================================

def main():

    print("=" * 60)
    print("STARTING DOCUMENT INGESTION")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"ERROR: Data folder not found: {DATA_DIR}")
        return

    total_added = 0

    # --------------------------------------------------------
    # Load every document
    # --------------------------------------------------------

    for file_path in DATA_DIR.iterdir():

        if not file_path.is_file():
            continue

        print(f"\nLoading: {file_path.name}")

        chunks = create_chunks_for_file(file_path)
        print(f"Chunks created from {file_path.name}: {len(chunks)}")
        if chunks:
            total_added += add_documents(chunks)

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STORING DOCUMENTS IN THE SELECTED VECTOR DATABASE")
    print("=" * 60)

    print(f"Total chunks stored: {total_added}")

    if not total_added:
        print("ERROR: No document chunks were created.")
        return

    print("=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
