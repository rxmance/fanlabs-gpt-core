import openai
import streamlit as st
import os
from dotenv import load_dotenv

# Load .env only for local testing (won't affect Streamlit Cloud)
load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = openai.OpenAI(api_key=api_key)