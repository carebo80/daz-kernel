# app/core/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "6"))
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q5_K_M")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL fehlt (.env)")
