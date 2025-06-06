import streamlit as st
import openai
import faiss

st.set_page_config(page_title="FanLabs GPT", layout="centered")

st.title("📚 Welcome to FanLabs GPT")
st.markdown("Choose a tool from the sidebar to get started.")

# ✅ Set OpenAI key from nested secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Load FAISS index and metadata (if needed here)
try:
    index = faiss.read_index("fanlabs_vector_index.faiss")
except Exception as e:
    st.error("Error loading FAISS index. Check that the file exists.")