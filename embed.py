
import openai
import faiss
import numpy as np
import json
import os
from pathlib import Path

# Load API key from environment or paste here directly (not recommended long term)
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Load the text chunks
with open("fanlabs_chunks.json", "r") as f:
    chunks = json.load(f)

# Embed using OpenAI text-embedding-3-small
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

# Create and populate FAISS index
dimension = 1536
index = faiss.IndexFlatL2(dimension)
metadata = []

for chunk in chunks:
    vector = np.array(get_embedding(chunk["text"]), dtype="float32")
    index.add(np.array([vector]))
    metadata.append(chunk)

# Save vector index
faiss.write_index(index, "fanlabs_vector_index.faiss")

# Save metadata
with open("fanlabs_chunk_metadata.json", "w") as f:
    json.dump(metadata, f)

print("✅ Embeddings complete. FAISS index and metadata saved.")
