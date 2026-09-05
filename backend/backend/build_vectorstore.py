import chromadb
from sentence_transformers import SentenceTransformer

# Import your existing chunking function
from chunking import create_chunks


# ----------------------------------------
# 1. Load embedding model
# ----------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------
# 2. Create persistent ChromaDB
# ----------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)
collection = client.get_or_create_collection(
    name="company_documents"
)

# ----------------------------------------
# 3. Get chunks from Task 5
# ----------------------------------------
chunks = create_chunks()
print("Total chunks received:", len(chunks))

# ----------------------------------------
# 4. Extract text
# ----------------------------------------
texts = [
    chunk.page_content
    for chunk in chunks
]

# ----------------------------------------
# 5. Generate embeddings
# ----------------------------------------
embeddings = model.encode(texts)
print(
    "Embedding vector length:",
    len(embeddings[0])
)

# ----------------------------------------
# 6. Prepare metadata
# ----------------------------------------
metadatas = []
for chunk in chunks:
    metadata = chunk.metadata.copy()
    # Chroma metadata must contain simple values
    #metadata = {
    key: str(value)
    for key, value in metadata.items()
   metadatas.append(metadata)

# ----------------------------------------
# 7. Create unique IDs
# ----------------------------------------
ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

# ----------------------------------------
# 8. Store everything in ChromaDB
# ----------------------------------------
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)
print("Successfully stored chunks:", collection.count())
print("ChromaDB ingestion completed!")Test-Path
# ----------------------------------------
# 9. Store everything in ChromaDB
# ----------------------------------------
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)
print("Successfully stored chunks:", collection.count())
print("ChromaDB ingestion completed!")Test-Path
import chromadb
from sentence_transformers import SentenceTransformer

# Import your existing chunking function
from chunking import create_chunks


# ----------------------------------------
# 1. Load embedding model
# ----------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------
# 2. Create persistent ChromaDB
# ----------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)
collection = client.get_or_create_collection(
    name="company_documents"
)

# ----------------------------------------
# 3. Get chunks from Task 5
# ----------------------------------------
chunks = create_chunks()
print("Total chunks received:", len(chunks))

# ----------------------------------------
# 4. Extract text
# ----------------------------------------
texts = [
    chunk.page_content
    for chunk in chunks
]
# ----------------------------------------
# 5. Generate embeddings
# ----------------------------------------
embeddings = model.encode(texts)
print(
    "Embedding vector length:",
    len(embeddings[0])
)

# ----------------------------------------
# 6. Prepare metadata
# ----------------------------------------
metadatas = []
for chunk in chunks:
    metadata = chunk.metadata.copy()
    # Chroma metadata must contain simple values
    #metadata = {
    key: str(value)
    for key, value in metadata.items()
   metadatas.append(metadata)

# ----------------------------------------
# 7. Create unique IDs
# ----------------------------------------
ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]
# ----------------------------------------
# 8. Store everything in ChromaDB
# ----------------------------------------
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)
print("Successfully stored chunks:", collection.count())
print("ChromaDB ingestion completed!")Test-Path
# ----------------------------------------
# 9. Store everything in ChromaDB
# ----------------------------------------
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)
print("Successfully stored chunks:", collection.count())
print("ChromaDB ingestion completed!")Test-Path
