from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import load_document


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    # Create the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    total_chunks = 0

    # Process every file in the data folder
    for file_path in DATA_DIR.iterdir():

        if not file_path.is_file():
            continue

        # Load the document using Task 4's function
        documents = load_document(file_path)

        if not documents:
            continue

        # Split the documents into chunks
        chunks = splitter.split_documents(documents)

        print("\n" + "=" * 60)
        print(f"File: {file_path.name}")
        print(f"Original documents/pages: {len(documents)}")
        print(f"Chunks generated: {len(chunks)}")
        print("=" * 60)

        # Show first 3 chunks for inspection
        for i, chunk in enumerate(chunks[:3], start=1):
            print(f"\nChunk {i}")
            print("-" * 60)
            print(f"Characters: {len(chunk.page_content)}")
            print(f"Source: {chunk.metadata.get('source')}")
            print(f"Page: {chunk.metadata.get('page')}")
            print(f"Content:\n{chunk.page_content[:500]}")

        total_chunks += len(chunks)

    print("\n" + "=" * 60)
    print(f"TOTAL CHUNKS GENERATED: {total_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    main()