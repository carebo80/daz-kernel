# app/services/embedder.py
import os
from sentence_transformers import SentenceTransformer

_MODEL = None

def get_embedder() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer(model_name)
    return _MODEL
