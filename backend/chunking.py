from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import load_all_documents


def create_chunks():

    # Create the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Load documents from Task 4
    documents = load_all_documents()

    # Split documents into chunks
    chunks = splitter.split_documents(documents)

    print("Total documents:", len(documents))
    print("Total chunks:", len(chunks))

    # Display first 5 chunks for manual review
    print("\n========== SAMPLE CHUNKS ==========\n")

    for i, chunk in enumerate(chunks[:5], start=1):

        print(f"--- Chunk {i} ---")
        print(chunk.page_content)
        print("\nMetadata:", chunk.metadata)
        print("\n")


if __name__ == "__main__":
    create_chunks()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import load_all_documents


def create_chunks():

    # Create the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Load documents from Task 4
    documents = load_all_documents()

    # Split documents into chunks
    chunks = splitter.split_documents(documents)

    print("Total documents:", len(documents))
    print("Total chunks:", len(chunks))

    # Display first 5 chunks for manual review
    print("\n========== SAMPLE CHUNKS ==========\n")

    for i, chunk in enumerate(chunks[:5], start=1):

        print(f"--- Chunk {i} ---")
        print(chunk.page_content)
        print("\nMetadata:", chunk.metadata)
        print("\n")


if __name__ == "__main__":
    create_chunks()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import load_all_documents


def create_chunks():

    # Create the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Load documents from Task 4
    documents = load_all_documents()

    # Split documents into chunks
    chunks = splitter.split_documents(documents)

    print("Total documents:", len(documents))
    print("Total chunks:", len(chunks))

    # Display first 5 chunks for manual review
    print("\n========== SAMPLE CHUNKS ==========\n")

    for i, chunk in enumerate(chunks[:5], start=1):

        print(f"--- Chunk {i} ---")
        print(chunk.page_content)
        print("\nMetadata:", chunk.metadata)
        print("\n")


if __name__ == "__main__":
    create_chunks()



