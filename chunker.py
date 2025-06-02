import os
import json

# Folder containing cleaned .txt files
folder_path = "Clean-knowledge"
chunk_size = 500  # You can adjust this as needed

chunks = []

def split_text(text, size):
    return [text[i:i+size] for i in range(0, len(text), size)]

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            full_text = f.read()
            text_chunks = split_text(full_text, chunk_size)
            for chunk in text_chunks:
                chunks.append({"source": filename, "text": chunk.strip()})

with open("fanlabs_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Chunked {len(chunks)} text blocks from files in '{folder_path}'.")
