import faiss
import json

def load_index_and_metadata():
    index = faiss.read_index("fanlabs_vector_index.faiss")
    with open("fanlabs_chunk_metadata.json", "r") as f:
        metadata = json.load(f)
    return index, metadata