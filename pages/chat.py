import streamlit as st
from utils.openai_client import client
from utils.prompts import base_system_prompt
from utils.faiss_helpers import load_index_and_metadata

from huggingface_hub import login
import os

# ✅ Correct Hugging Face token from nested secrets
hf_token = st.secrets["huggingface"]["token"] or os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)

from sentence_transformers import SentenceTransformer
import numpy as np

st.title("💬 FanLabs GPT")

user_query = st.text_input("Ask a question about your data")

if user_query:
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([user_query])
    
    D, I = index.search(np.array(query_embedding).astype("float32"), k=5)

    retrieved_chunks = [metadata[i]["text"] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": base_system_prompt},
        {"role": "user", "content": f"{context}\n\n{user_query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    st.markdown("### Response")
    st.write(response.choices[0].message.content)