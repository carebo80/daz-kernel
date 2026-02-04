import os
import requests

def ollama_generate(model: str, prompt: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "top_p": 0.9}}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")