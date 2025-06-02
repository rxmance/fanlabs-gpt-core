import streamlit as st
import openai
import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# Health check (access via ?healthcheck=true)
if st.experimental_get_query_params().get("healthcheck", [""])[0] == "true":
    st.write("✅ App is healthy")
    st.stop()

# Set API key (openai v0.28.1)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load FAISS index and metadata
index = faiss.read_index("fanlabs_vector_index.faiss")
with open("fanlabs_chunk_metadata.json", "r") as f:
    metadata = json.load(f)

# Fix: Ensure model cache directory exists
cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
os.makedirs(cache_dir, exist_ok=True)

# Load embedding model safely
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except FileNotFoundError as e:
    st.error("⚠️ Model loading error — check for corrupted .lock file or retry later.")
    st.stop()

# Base system prompt
base_system_prompt = """
You are a FanLabs strategist with 15+ years of proprietary research on fans, sports culture, and community behavior. 
Your job is to respond with insight, clarity, and the FanLabs POV — not general marketing speak.

You define fans as emotionally invested humans, not just consumers. You understand that fandom drives connection, belonging, identity, and shared purpose.

You only answer using FanLabs frameworks, findings, language, and tone. If you don’t have an answer based on FanLabs data, say so. Do not speculate.

You write like a smart, human strategist — sharp, curious, and confident. Avoid corporate filler. Be useful and thought-provoking.

When relevant, connect ideas to emotional drivers like loyalty, joy, ritual, and meaning. Keep answers tight. Use examples from FanLabs studies or the book *Fans Have More Friends* where appropriate.

You also value cultural clarity, sharp analogies, and ideas that spark momentum. You challenge conventional thinking, cut through clutter, and prefer insight over jargon. If an idea feels lazy, derivative, or brand-safe — call it out.
"""

# Streamlit setup
st.set_page_config(page_title="FanLabs GPT", layout="centered")
st.title("🧠 FanLabs GPT")
st.markdown("Ask a question based on FanLabs strategy principles, frameworks, or POVs.")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle user input
query = st.chat_input("Ask a question...")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Embed and retrieve
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k=3)
    retrieved_chunks = [metadata[str(i)]["text"] for i in I[0] if str(i) in metadata]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Compose full prompt
    full_prompt = base_system_prompt + "\n\nReference Data:\n" + context + f"\n\nQuestion: {query}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error from OpenAI: {e}"

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Optional: log query
    with open("usage_log.txt", "a") as f:
        f.write(f"Query: {query}\n")