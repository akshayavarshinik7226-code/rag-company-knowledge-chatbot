from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test sentences
sentences = [
    "Employees are entitled to annual leave.",
    "Employees can take vacation days.",
    "The company provides technical support for laptops."
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

# Display results
print("Number of sentences:", len(embeddings))
print("Vector length:", len(embeddings[0]))
print("First embedding:")
print(embeddings[0])