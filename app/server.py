import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
import psycopg2
from sentence_transformers import SentenceTransformer
import traceback
from pgvector.psycopg2 import register_vector
from pgvector import Vector
import json
import requests
from datetime import date
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "6"))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL fehlt (.env)")

app = FastAPI(title="DaZ Kernel (RAG)")
templates = Jinja2Templates(directory="templates")
embedder = SentenceTransformer(MODEL_NAME)

def db():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn
def ollama_generate(model: str, prompt: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")

class SearchHit(BaseModel):
    id: int
    source: str
    page: Optional[int] = None
    chunk_index: int
    score: float
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.get("/stats")
def stats():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM doc_chunks;")
            n = cur.fetchone()[0]
            cur.execute("SELECT source, COUNT(*) FROM doc_chunks GROUP BY source ORDER BY 2 DESC;")
            sources = [{"source": r[0], "count": r[1]} for r in cur.fetchall()]
    return {"count": n, "sources": sources, "db": DATABASE_URL}

@app.get("/search", response_model=List[SearchHit])
def search(q: str = Query(..., min_length=2), top_k: int = TOP_K_DEFAULT):
    q_emb = embedder.encode([q], normalize_embeddings=True)[0].tolist()
    q_vec = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    sql = """
    SELECT id, source, page, chunk_index, content,
           1 - (embedding <=> vector(%s)) AS score
    FROM doc_chunks
    ORDER BY embedding <=> vector(%s)
    LIMIT %s;
    """

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (q_vec, q_vec, top_k))
            rows = cur.fetchall()

    # DEBUG: nie stillschweigend []
    print("SEARCH DEBUG q=", q, "rows=", len(rows))

    return [
        SearchHit(
            id=r[0], source=r[1], page=r[2], chunk_index=r[3],
            content=r[4], score=float(r[5])
        )
        for r in rows
    ]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q_emb = embedder.encode([req.question], normalize_embeddings=True)[0].tolist()
    q_vec = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    sql = """
    SELECT id, source, page, chunk_index, content,
           1 - (embedding <=> vector(%s)) AS score
    FROM doc_chunks
    ORDER BY embedding <=> vector(%s)
    LIMIT %s;
    """

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (q_vec, q_vec, req.top_k))
            rows = cur.fetchall()

    # rows immer ausgeben, auch wenn scores niedrig sind
    if not rows:
        return AskResponse(
            answer="DB enthält keine Chunks (oder Query fehlgeschlagen).",
            citations=[]
        )

    citations = []
    snippets = []
    for r in rows:
        _id, source, page, chunk_index, content, score = r
        citations.append({
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "score": float(score)
        })
        snippet = (content[:320] + "…") if len(content) > 320 else content
        snippets.append(f"- {source}, S. {page} (Score {float(score):.3f})\n  » {snippet}")

    answer = (
        f"Top-{len(rows)} Belegstellen zu deiner Frage:\n"
        f"„{req.question}“\n\n"
        + "\n\n".join(snippets)
        + "\n\nNächster Schritt: Sobald du die offiziellen GeR-Deskriptoren (z.B. Globalskala/Selbstbeurteilung oder Companion Volume) als PDF dazulegst, werden die Treffer für A2 deutlich präziser."
    )

    return AskResponse(answer=answer, citations=citations)
class AskHybridRequest(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT
    text_terms: Optional[List[str]] = None  # optionale harte Suchbegriffe
class PlanUnitRequest(BaseModel):
    topic: str = "Bank"
    level: str = "A2"
    strong_group: bool = True
    time_start: str = "08:45"
    time_end: str = "11:15"
    top_k: int = TOP_K_DEFAULT
    text_terms: Optional[List[str]] = None
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class PlanUnitResponse(BaseModel):
    unit_title: str
    meta: Dict[str, Any]
    ger: Dict[str, Any]
    language_support: Dict[str, Any]
    phases: List[Dict[str, Any]]
    materials: List[Dict[str, Any]]

@app.post("/ask_hybrid", response_model=AskResponse)
def ask_hybrid(req: AskHybridRequest):
    q = req.question
    top_k = req.top_k

    # 1) Text Terms bestimmen (wenn nicht geliefert)
    # Für GeR/CEFR: ein paar robuste Anker
    default_terms = [
        "A2",
        "routinem",
        "Kontaktgespr",
        "Austausch von Informationen",
        "verständigen",
    ]
    terms = req.text_terms or default_terms

    # 2) TEXT SEARCH (ILIKE)
    text_hits = []
    with db() as conn:
        with conn.cursor() as cur:
            for t in terms:
                cur.execute(
                    """
                    SELECT id, source, page, chunk_index, content
                    FROM doc_chunks
                    WHERE content ILIKE %s
                    ORDER BY id
                    LIMIT %s;
                    """,
                    (f"%{t}%", top_k),
                )
                for r in cur.fetchall():
                    text_hits.append(r)

    # Dedup by (id)
    seen = set()
    dedup_text_hits = []
    for r in text_hits:
        if r[0] in seen:
            continue
        seen.add(r[0])
        dedup_text_hits.append(r)

    # 3) (Optional) VECTOR SEARCH – nur falls du es willst / es liefert
    # Wir versuchen es, aber wenn es leer ist, ist das ok.
    vector_hits = []
    try:
        vector_hits = search(q, top_k)  # nutzt deinen /search code
    except Exception:
        vector_hits = []

    # 4) Antwort bauen
    citations = []
    parts = []

    # Textbelege zuerst (GeR-Normsatz / Globalskala etc.)
    if dedup_text_hits:
        parts.append("Texttreffer (exakte Stellen):")
        for (id_, source, page, chunk_index, content) in dedup_text_hits[:top_k]:
            snippet = (content[:280] + "…") if len(content) > 280 else content
            parts.append(f"- {source}, S. {page} (Chunk {chunk_index})\n  » {snippet}")
            citations.append({
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "score": 1.0
            })

    # Vektorbelege danach (handlungsnahe Skalen / Kontext)
    if vector_hits:
        parts.append("\nSemantische Treffer (Kontext/Operationalisierung):")
        for h in vector_hits[:top_k]:
            snippet = (h.content[:280] + "…") if len(h.content) > 280 else h.content
            parts.append(f"- {h.source}, S. {h.page} (Score {h.score:.3f})\n  » {snippet}")
            citations.append({
                "source": h.source,
                "page": h.page,
                "chunk_index": h.chunk_index,
                "score": h.score
            })

    if not parts:
        return AskResponse(
            answer="Keine Treffer gefunden. Tipp: andere Suchbegriffe in text_terms mitgeben (z.B. 'A2', 'routinem', 'Kontaktgespr').",
            citations=[]
        )

    # Didaktischer Kurztext vorneweg (pragmatisch, ohne LLM)
    answer = (
        f"Frage: {q}\n\n"
        "Kurzbezug GeR (Belegstellen):\n"
        "- Die Lernziele können an A2-Deskriptoren festgemacht werden (routinemässige Situationen / direkter Informationsaustausch / kurze Kontaktgespräche).\n"
        "- Für die Unterrichtsplanung werden diese Deskriptoren in konkrete Handlungsaufgaben übersetzt (z.B. Fragen stellen, Informationen austauschen, klären/nachfragen).\n\n"
        + "\n\n".join(parts)
    )

    return AskResponse(answer=answer, citations=citations)

@app.get("/search_dbseed")
def search_dbseed(top_k: int = 5):
    sql = """
    WITH q AS (
      SELECT embedding AS v
      FROM doc_chunks
      WHERE embedding IS NOT NULL
      LIMIT 1
    )
    SELECT id, source, page, chunk_index,
           1 - (embedding <=> (SELECT v FROM q)) AS score
    FROM doc_chunks
    ORDER BY embedding <=> (SELECT v FROM q)
    LIMIT %s;
    """
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (top_k,))
            rows = cur.fetchall()
            print("SEARCH rows:", len(rows))

    return [{"id": r[0], "source": r[1], "page": r[2], "chunk_index": r[3], "score": float(r[4])} for r in rows]
@app.get("/search_text")
def search_text(q: str, top_k: int = 5):
    sql = """
    SELECT id, source, page, chunk_index, content
    FROM doc_chunks
    WHERE content ILIKE %s
    LIMIT %s;
    """
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (f"%{q}%", top_k))
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "source": r[1],
            "page": r[2],
            "chunk_index": r[3],
            "content": r[4][:400]
        }
        for r in rows
    ]
@app.post("/plan_unit", response_model=PlanUnitResponse)
def plan_unit(req: PlanUnitRequest):
    # 1) GeR-Belege holen (hybrid, belegbar)
    hybrid_req = AskHybridRequest(
        question=f"{req.level} {req.topic} Gespräch",
        top_k=req.top_k,
        text_terms=req.text_terms or ["A2", "routinem", "Kontaktgespr", "Austausch", "Fragen"]
    )
    ger_resp = ask_hybrid(hybrid_req)

    # 2) Lokale AI: Sprachmittel generieren (nur Sprachmittel, kein GeR)
    # Wir erzwingen JSON-Output.
    prompt = f"""
    Du bist DaZ-Lehrmittelautor:in (A2, Schweiz).
    Erstelle NUR Sprachmittel für einen {req.level}-Kurs, Thema: {req.topic} (Bank).

    Didaktische Regeln:
    - Niveau: A2 (stark), Alltagssprache
    - Erlaubt: einfache Hauptsätze, Modalverben (möchte, kann), W-Fragen
    - Verboten: Nebensätze mit "weil/obwohl", Konjunktiv II, Fachsprache
    - Ziel: mündliche Interaktion (Kontaktgespräch, Informationsaustausch)

    VERPFLICHTEND:
    - Die Ausgabe MUSS genau 2 Mini-Dialoge enthalten.
    - mini_dialogues darf NICHT leer sein.

    Gib außerdem:
    - 10–15 Wortschatz-Einträge
    - 6–10 Redemittel
    - 1–2 Grammatik-Foki

    Gib als EINZIGES Ergebnis gültiges JSON zurück (kein Markdown, keine Erklärungen).

    JSON-Schema:
    {{
    "vocabulary": [{{"word": "...", "note": "kurz"}}],
    "phrases": [{{"de": "...", "function": "z.B. begrüssen/nachfragen/bitte"}}],
    "grammar_focus": [{{"topic": "...", "examples": ["...","..."]}}],
    "mini_dialogues": [
        {{
        "title":"...",
        "lines":[
            {{"role":"Kundin/Kunde","text":"..."}},
            {{"role":"Bank","text":"..."}}
        ]
        }}
    ]
    }}

    Thema-Fokus:
    - Konto eröffnen
    - Geld einzahlen / abheben
    - Ausweis, Formular, Unterschrift
    """
    try:
        lm_raw = ollama_generate(req.ollama_model, prompt)
        language_support = json.loads(lm_raw)
    except Exception as e:
        # Fallback ohne AI (damit Endpoint nie bricht)
        language_support = {
            "vocabulary": [{"word": "Konto", "note": "bei der Bank"}, {"word": "Ausweis", "note": "ID/Pass"}],
            "phrases": [{"de": "Ich möchte ein Konto eröffnen.", "function": "Wunsch äußern"}],
            "grammar_focus": [{"topic": "W-Fragen / Ja-Nein-Fragen", "examples": ["Was brauche ich?", "Haben Sie einen Ausweis?"]}],
            "mini_dialogues": []
        }

    # 3) Phasenplan (didaktische Phasen, keine Sozialformen)
    # Zeitfenster 08:45–11:15 = 150 Min (mit Pause)
    phases = [
        {
            "phase": "Ankommen & Aktivierung",
            "minutes": 10,
            "aim": "Vorwissen aktivieren, Setting Bank einführen",
            "activity": "Kurzer Einstieg: 'Wann wart ihr zuletzt bei der Bank?' + 5 Schlüsselwörter an Tafel."
        },
        {
            "phase": "Input & Modellierung",
            "minutes": 20,
            "aim": "Redemittel/Schlüsselstrukturen bereitstellen",
            "activity": "Lehrperson modelliert 2 Mini-Dialoge (Konto eröffnen / Geld abheben) mit Fokus auf Phrasen."
        },
        {
            "phase": "Gelenkte Übung",
            "minutes": 25,
            "aim": "Fragen/Antworten automatisieren",
            "activity": "Dialog-Bausteine ordnen + Lückendialoge (W-Fragen, 'ich möchte', Höflichkeit)."
        },
        {
            "phase": "Anwendung: Rollenspiel 1",
            "minutes": 25,
            "aim": "Handlungsfähigkeit: Informationsaustausch im Banksetting",
            "activity": "Rollenspiel mit Rollenkarte: Kund:in will Konto eröffnen, Bank stellt Fragen (Name, Adresse, Ausweis)."
        },
        {
            "phase": "Pause",
            "minutes": 10,
            "aim": "Erholung",
            "activity": "Pause"
        },
        {
            "phase": "Anwendung: Rollenspiel 2",
            "minutes": 25,
            "aim": "Transfer: Variation & spontane Reaktion",
            "activity": "Neue Situation: Geld einzahlen/abheben + Problem (Karte vergessen / PIN falsch) – mit Nachfragen/klären."
        },
        {
            "phase": "Auswertung & Sprachfokus",
            "minutes": 20,
            "aim": "Fehler sammeln, Redemittel festigen",
            "activity": "Gemeinsame Auswertung: 5 'gute Sätze' + 3 typische Fehler → kurze Korrektur/Drill."
        },
        {
            "phase": "Abschluss & Mini-Check",
            "minutes": 15,
            "aim": "Selbstcheck A2: 'Kann ich…?'",
            "activity": "Mini-Checkliste + 2 neue Sätze schriftlich produzieren (z.B. Formularfelder ausfüllen)."
        }
    ]

    # 4) Materialienliste (was du danach automatisieren kannst)
    materials = [
        {"type": "Rollenkarten", "items": ["Konto eröffnen", "Geld abheben", "Problemfall: Karte/PIN"]},
        {"type": "Dialogstreifen", "items": ["Begrüßung", "Wunsch äußern", "Nachfragen", "Abschluss"]},
        {"type": "Wortschatzblatt", "items": ["Bank-Wörter (A2)", "Verben: eröffnen/abheben/einzahlen", "Höflichkeit"]},
        {"type": "Mini-Formular", "items": ["Name, Adresse, Geburtsdatum, Ausweisnummer (Übung)"]}
    ]

    # 5) GeR-Auszug kompakt
    ger = {
        "answer": ger_resp.answer,
        "citations": ger_resp.citations
    }

    title = f"{req.level} – {req.topic}: Bankgespräch (08:45–11:15)"

    return PlanUnitResponse(
        unit_title=title,
        meta={
            "date": str(date.today()),
            "level": req.level,
            "topic": req.topic,
            "time_start": req.time_start,
            "time_end": req.time_end,
            "strong_group": req.strong_group,
            "ollama_model": req.ollama_model
        },
        ger=ger,
        language_support=language_support,
        phases=phases,
        materials=materials
    )
    # ---- Sicherstellen, dass 2 Mini-Dialoge existieren ----
    if not language_support.get("mini_dialogues"):
        language_support["mini_dialogues"] = [
            {
                "title": "Konto eröffnen",
                "lines": [
                    {"role": "Kundin/Kunde", "text": "Guten Tag. Ich möchte ein Konto eröffnen."},
                    {"role": "Bank", "text": "Guten Tag. Haben Sie einen Ausweis?"},
                    {"role": "Kundin/Kunde", "text": "Ja, hier ist mein Ausweis."},
                    {"role": "Bank", "text": "Danke. Bitte füllen Sie dieses Formular aus."}
                ]
            },
            {
                "title": "Geld abheben",
                "lines": [
                    {"role": "Kundin/Kunde", "text": "Ich möchte Geld abheben."},
                    {"role": "Bank", "text": "Wie viel Geld möchten Sie abheben?"},
                    {"role": "Kundin/Kunde", "text": "Ich möchte 200 Franken abheben."},
                    {"role": "Bank", "text": "Bitte geben Sie Ihre Karte ein."}
                ]
            }
        ]
@app.get("/preview", response_class=HTMLResponse)
def preview_get(request: Request):
    form = {
        "topic": "Bank",
        "level": "A2",
        "time_start": "08:45",
        "time_end": "11:15",
        "text_terms": "A2,routinem,Kontaktgespr,Austausch,Fragen",
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "top_k": "5",
        "strong_group": "true",
    }
    return templates.TemplateResponse("preview.html", {"request": request, "form": form, "unit": None})


@app.post("/preview", response_class=HTMLResponse)
def preview_post(
    request: Request,
    topic: str = Form("Bank"),
    level: str = Form("A2"),
    time_start: str = Form("08:45"),
    time_end: str = Form("11:15"),
    text_terms: str = Form("A2,routinem,Kontaktgespr,Austausch,Fragen"),
    ollama_model: str = Form("llama3.1:8b"),
    top_k: str = Form("5"),
    strong_group: str = Form("true"),
):
    terms = [t.strip() for t in text_terms.split(",") if t.strip()]
    req = PlanUnitRequest(
        topic=topic,
        level=level,
        time_start=time_start,
        time_end=time_end,
        strong_group=(strong_group.lower() == "true"),
        top_k=int(top_k),
        text_terms=terms,
        ollama_model=ollama_model,
    )
    unit = plan_unit(req)
    form = {
        "topic": topic,
        "level": level,
        "time_start": time_start,
        "time_end": time_end,
        "text_terms": text_terms,
        "ollama_model": ollama_model,
        "top_k": top_k,
        "strong_group": strong_group,
    }
    return templates.TemplateResponse("preview.html", {"request": request, "form": form, "unit": unit})
