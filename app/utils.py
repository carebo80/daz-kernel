from datetime import datetime
import json
import re


def parse_json_loose(s: str) -> dict:
    # 1) Direkt versuchen
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Falls ```json ... ``` drin ist
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    # 3) Sonst: erste { ... } “greedy” herausziehen
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Kein JSON-Objekt in Ollama-Response gefunden")

def minutes_between(t1: str, t2: str) -> int:
    fmt = "%H:%M"
    a = datetime.strptime(t1, fmt)
    b = datetime.strptime(t2, fmt)
    return int((b - a).total_seconds() // 60)

def slugify(s: str) -> str:
    s = s.strip().lower()
    # sehr simple slugify, reicht für jetzt
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in [" ", "-", "_", "/"]:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")