import streamlit as st
from dotenv import load_dotenv
import os
import faiss
import json
from sentence_transformers import SentenceTransformer
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index and metadata
index = faiss.read_index("fanlabs_vector_index.faiss")
with open("fanlabs_chunk_metadata.json", "r") as f:
    metadata = json.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# System prompt for FanLabs GPT
base_system_prompt = """
You are a FanLabs strategist with 15+ years of proprietary research on fans, sports culture, and community behavior...
(keep your full prompt here)
"""

# Streamlit UI setup
st.set_page_config(page_title="FanLabs GPT", layout="centered")
st.title("🧠 FanLabs GPT")
st.markdown("Ask a question based on FanLabs strategy principles, frameworks, or POVs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
query = st.chat_input("Ask a question...")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Embed the query and retrieve relevant context
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k=3)
    retrieved_chunks = [metadata[str(i)]["text"] for i in I[0] if str(i) in metadata]

    context = "\n\n---\n\n".join(retrieved_chunks)
    full_prompt = base_system_prompt + "\n\nReference Data:\n" + context + f"\n\nQuestion: {query}"

    # Make OpenAI API call (correct usage for 1.25.1+)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error from OpenAI: {e}"

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})