import streamlit as st
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import faiss
import pickle
import json

from sentence_transformers import SentenceTransformer

# Load .env variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index and metadata
index = faiss.read_index("fanlabs_vector_index.faiss")
with open("fanlabs_chunk_metadata.json", "r") as f:
    metadata = json.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("🧠 FanLabs GPT")
st.markdown("Ask me anything about your docs!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
query = st.chat_input("Ask a question...")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Embed query and search FAISS
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k=3)

    retrieved_chunks = [metadata[str(i)]["text"] for i in I[0] if str(i) in metadata]

    prompt = (
        "You are a helpful assistant. Answer the question based only on the following context:\n\n"
        + "\n\n---\n\n".join(retrieved_chunks)
        + f"\n\nQuestion: {query}"
    )

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error from OpenAI: {e}"

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
