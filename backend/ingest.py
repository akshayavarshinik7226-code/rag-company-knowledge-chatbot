from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_document(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        return []

    return loader.load()


def main():
    for file_path in DATA_DIR.iterdir():
        if not file_path.is_file():
            continue

        documents = load_document(file_path)

        print(f"\nFile: {file_path.name}")
        print(f"Number of documents/pages: {len(documents)}")

        for document in documents:
            print(f"Characters: {len(document.page_content)}")
            print(f"Source: {document.metadata.get('source')}")
            print(f"Preview: {document.page_content[:200]}")
            print("-" * 50)


if __name__ == "__main__":
    main()