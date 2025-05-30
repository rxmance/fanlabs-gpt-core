from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

# Load your chunks
with open("fanlabs_chunks.json", "r") as f:
    chunks = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = []
metadata = []

# Embed each chunk
for i, chunk in enumerate(chunks):
    vector = model.encode(chunk["text"])
    vectors.append(vector)
    metadata.append({"id": i, "text": chunk["text"]})

# Convert to FAISS index
index = faiss.IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors).astype("float32"))

# Save index + metadata
faiss.write_index(index, "fanlabs_vector_index.faiss")
with open("fanlabs_chunk_metadata.json", "w") as f:
    json.dump(metadata, f)

print("✅ SentenceTransformer embeddings complete. Index and metadata saved.")