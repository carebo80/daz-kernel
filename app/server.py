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
import json, re
import uuid
import requests
from datetime import date, datetime
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi import Body
from app.prompts.language_support import build_language_support_prompt
from app.phases import build_phases

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
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "top_p": 0.9}}
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

class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
class UnitCitationIn(BaseModel):
    chunk_id: int
    score: float = 0.0
    quote: str = ""

class UnitCreateRequest(BaseModel):
    level: str                 # "A2"
    topic: Optional[str] = None  # "Bank" (Titel)
    topic_slug: Optional[str] = None  # "bank" (optional; wenn None -> aus topic abgeleitet)
    time_start: str = ""
    time_end: str = ""
    strong_group: bool = False
    title: str = ""
    notes: str = ""
    plan: Dict[str, Any] = {}              # Feinplanung JSON
    language_support: Dict[str, Any] = {}  # vocabulary/phrases/grammar_focus/mini_dialogues JSON
    citations: List[UnitCitationIn] = []   # optional

class UnitCitationOut(BaseModel):
    id: str
    chunk_id: int
    score: float
    quote: str
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None

class UnitResponse(BaseModel):
    id: str
    created_at: str
    updated_at: str
    level: str
    topic: Optional[str] = None
    time_start: str
    time_end: str
    strong_group: bool
    title: str
    notes: str
    plan: Dict[str, Any]
    language_support: Dict[str, Any]
    citations: List[UnitCitationOut] = []

class TopicOut(BaseModel):
    id: str
    slug: str
    title: str

class TopicWithUnitsOut(BaseModel):
    id: str
    slug: str
    title: str
    unit_count: int

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

def find_chunk_id_by_quote(conn, quote: str) -> Optional[int]:
    if not quote or len(quote.strip()) < 20:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM doc_chunks
            WHERE content ILIKE %s
            ORDER BY id
            LIMIT 1
            """,
            (f"%{quote[:120]}%",)  # nur kurzer Teil reicht
        )
        row = cur.fetchone()
        return int(row[0]) if row else None

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

@app.get("/topics", response_model=List[TopicOut])
def list_topics(q: Optional[str] = None, limit: int = 200):
    """
    Listet Topics aus p_topics.
    Optional: q=... filtert via ILIKE auf slug/title.
    """
    sql = """
    SELECT id, slug, title
    FROM p_topics
    WHERE (%s IS NULL OR title ILIKE %s OR slug ILIKE %s)
    ORDER BY title ASC
    LIMIT %s;
    """
    with db() as conn:
        with conn.cursor() as cur:
            like = f"%{q}%" if q else None
            cur.execute(sql, (q, like, like, limit))
            rows = cur.fetchall()

    return [
    TopicOut(id=str(r[0]), slug=r[1], title=r[2])
    for r in rows
]

@app.get("/search", response_model=List[SearchHit])
def search(
    q: str = Query(..., min_length=2),
    top_k: int = TOP_K_DEFAULT,
    source_like: Optional[str] = None,   # neu
):
    q_emb = embedder.encode([q], normalize_embeddings=True)[0].tolist()
    qv = Vector(q_emb)

    sql = """
    SELECT id, source, page, chunk_index, content,
           1 - (embedding <=> %s) AS score
    FROM doc_chunks
    WHERE (%s IS NULL OR source ILIKE %s)
    ORDER BY embedding <=> %s
    LIMIT %s;
    """

    with db() as conn:
        with conn.cursor() as cur:
            like = f"%{source_like}%" if source_like else None
            cur.execute(sql, (qv, source_like, like, qv, top_k))
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

    # einfache Hybrid-Variante: Kandidaten via ILIKE, dann vector-rerank
    terms = ["routinem", "Kontaktgespr", "Austausch", "Information", "Deskriptor", "Globalskala", "Referenz"]
    likes = [f"%{t}%" for t in terms]

    level = "A2"  # später aus req ableiten

    sql = """
    WITH candidates AS (
    SELECT id, source, page, chunk_index, content, embedding
    FROM doc_chunks
    WHERE
        content ILIKE %(level_like)s
        AND (
        content ILIKE ANY(%(likes)s)
        OR source ILIKE '%%GeR%%'
        OR source ILIKE '%%Deskriptor%%'
        OR source ILIKE '%%Globalskala%%'
        )
    LIMIT 300
    )
    SELECT id, source, page, chunk_index, content,
        1 - (embedding <=> vector(%(q_vec)s)) AS score
    FROM candidates
    ORDER BY embedding <=> vector(%(q_vec)s)
    LIMIT %(top_k)s;
    """
    with db() as conn:
        with conn.cursor() as cur:
            print("SQL placeholders:", sql.count("%s"))
            print("PARAMS:", len((f"%{level}%", likes, q_vec, q_vec, req.top_k)))
            params = {
            "level_like": f"%{level}%",
            "likes": likes,
            "q_vec": q_vec,
            "top_k": req.top_k,
            }
            cur.execute(sql, params)
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
    level: Optional[str] = None
    top_k: int = TOP_K_DEFAULT
    text_terms: Optional[List[str]] = None

class PlanUnitRequest(BaseModel):
    topic: str
    level: str
    strong_group: bool = True
    time_start: str = "08:45"
    time_end: str = "11:15"
    top_k: int = TOP_K_DEFAULT
    text_terms: Optional[List[str]] = None
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    phase_model: str = "rita"

class PlanUnitResponse(BaseModel):
    unit_title: str
    meta: Dict[str, Any]
    ger: Dict[str, Any]
    language_support: Dict[str, Any]
    phases: List[Dict[str, Any]]
    materials: List[Dict[str, Any]]

class UnitFromPlanUnitRequest(BaseModel):
    topic: str = "Bank"
    level: str = "A2"
    time_start: str = "08:45"
    time_end: str = "11:15"
    strong_group: bool = True
    top_k: int = TOP_K_DEFAULT
    text_terms: Optional[List[str]] = None
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    title: Optional[str] = None
    notes: str = ""

class UnitFromPlanUnitResponse(BaseModel):
    unit: UnitResponse
    preview_url: str
    citations_saved: int

@app.post("/ask_hybrid", response_model=AskResponse)
def ask_hybrid(req: AskHybridRequest):
    q = req.question
    top_k = req.top_k
    lvl = req.level or "A1"
    # 1) Text Terms bestimmen (wenn nicht geliefert)
    # Für GeR/CEFR: ein paar robuste Anker
    default_terms = [
    lvl,
    "routinem", "Kontaktgespr", "Austausch", "Fragen"
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

    topic = req.topic.strip()
    level = req.level.strip().upper()
    strong_group = bool(req.strong_group)
    phase_model = getattr(req, "phase_model", "rita")

    duration = minutes_between(req.time_start, req.time_end)
    context = getattr(req, "context", None) or "Erwachsenenbildung, DaZ, Schweiz"

    phases = build_phases(
        topic=topic,
        level=level,
        strong_group=strong_group,
        duration=duration,
        model=phase_model,
    )

    hybrid_req = AskHybridRequest(
        question=f"{level} {topic} Interaktion / Gespräch",
        level=level,
        top_k=req.top_k,
        text_terms=req.text_terms or [level, topic, "Kontaktgespr", "Austausch", "Fragen"]
    )
    ger_resp = ask_hybrid(hybrid_req)

    if strong_group:
        group_desc = "starke Gruppe: schneller, mehr Variation, etwas komplexere Sätze"
        complexity_rules = """
- Nutze auch längere einfache Sätze.
- Erlaube kleine Erweiterungen (Adverbien, Ergänzungen).
- Baue kleine Transfer-Aufgaben ein.
"""
    else:
        group_desc = "unterstützungsbedürftige Gruppe: langsam, sehr kontrolliert, kurze Sätze"
        complexity_rules = """
- Nutze sehr kurze, klare Sätze.
- Wiederhole Strukturen.
- Vermeide Varianten.
- Nutze feste Satzmuster.
"""

    prompt = build_language_support_prompt(
        topic=topic,
        level=level,
        strong_group=strong_group,
        duration=duration,
        context=context,
        group_desc=group_desc,
        complexity_rules=complexity_rules,
        ger_answer=ger_resp.answer,
        ger_citations=ger_resp.citations,
        debug=getattr(req, "debug", False),
    )

    ollama_ok = False
    ollama_error = None

    # neutraler fallback
    language_support = {
        "vocabulary": [{"word": topic, "note": "Thema"}],
        "phrases": [{"de": "Guten Tag.", "function": "begrüssen"}],
        "grammar_focus": [{"topic": "W-Fragen", "examples": ["Was ist das?"]}],
        "mini_dialogues": [
            {"title": f"{topic}: Mini-Dialog 1", "lines": [
                {"role": "A", "text": "Guten Tag."},
                {"role": "B", "text": "Guten Tag. Wie geht es Ihnen?"},
                {"role": "A", "text": "Mir geht’s gut, danke. Und Ihnen?"},
                {"role": "B", "text": "Auch gut, danke."},
            ]},
            {"title": f"{topic}: Mini-Dialog 2", "lines": [
                {"role": "A", "text": f"Ich habe eine Frage zu {topic}."},
                {"role": "B", "text": "Ja, gern. Was möchten Sie wissen?"},
            ]},
        ],
    }

    try:
        lm_raw = ollama_generate(req.ollama_model, prompt)
        parsed = parse_json_loose(lm_raw)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama JSON ist kein Objekt (dict).")
        language_support = parsed
        ollama_ok = True
    except Exception as e:
        ollama_error = f"{type(e).__name__}: {e}"

    md = language_support.get("mini_dialogues") or []
    if len(md) != 2:
        # dein bestehender 2-dialog fallback
        language_support["mini_dialogues"] = [
            {
                "title": f"{topic}: Kontaktgespräch",
                "lines": [
                    {"role": "A", "text": f"Guten Tag. Ich habe eine Frage zum Thema {topic}."},
                    {"role": "B", "text": "Guten Tag. Ja, gern. Was möchten Sie wissen?"},
                    {"role": "A", "text": "Können Sie mir bitte helfen?"},
                    {"role": "B", "text": "Ja. Wir machen das zusammen."},
                ],
            },
            {
                "title": f"{topic}: Termin/Information",
                "lines": [
                    {"role": "A", "text": f"Ich brauche Informationen zu {topic}."},
                    {"role": "B", "text": "Okay. Was genau brauchen Sie?"},
                    {"role": "A", "text": "Was muss ich machen?"},
                    {"role": "B", "text": "Sie füllen ein Formular aus und bringen die Unterlagen mit."},
                ],
            },
        ]

    materials = [
        {"type": "Rollenkarten", "items": [f"Situation zu {topic}", "Nachfragen & Klären", "Problemfall (Missverständnis)"]},
        {"type": "Dialogstreifen", "items": ["Begrüssung", "Wunsch äussern", "Nachfragen", "Abschluss"]},
        {"type": "Wortschatzblatt", "items": [f"Wörter zu {topic} ({level})", "wichtige Verben", "Höflichkeit"]},
        {"type": "Mini-Formular", "items": ["Name, Kontakt, Notizen (Übung)"]},
    ]

    title = f"{level} – {topic}: Unterrichtseinheit ({req.time_start}–{req.time_end})"

    return PlanUnitResponse(
        unit_title=title,
        meta={
            "date": str(date.today()),
            "level": level,
            "topic": topic,
            "time_start": req.time_start,
            "time_end": req.time_end,
            "strong_group": strong_group,
            "phase_model": phase_model,
            "ollama_model": req.ollama_model,
            "ollama_ok": ollama_ok,
            "ollama_error": ollama_error,
        },
        ger={"answer": ger_resp.answer, "citations": ger_resp.citations},
        language_support=language_support,
        phases=phases,
        materials=materials,
    )

@app.get("/preview", response_class=HTMLResponse)
def preview_get(request: Request):
    # topics laden
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT slug, title FROM p_topics ORDER BY title")
            topics = cur.fetchall()

    form = {
        "topic": "bank",  # slug!
        "level": "A2",
        "time_start": "08:45",
        "time_end": "11:15",
        "text_terms": "A2,routinem,Kontaktgespr,Austausch,Fragen",
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "top_k": "5",
        "strong_group": "true",
    }

    return templates.TemplateResponse(
        "preview.html",
        {"request": request, "form": form, "unit": None, "topics": topics}
    )

@app.post("/preview", response_class=HTMLResponse)
def preview_post(
    request: Request,
    topic: str = Form("bank"),  # slug
    level: str = Form("A2"),
    time_start: str = Form("08:45"),
    time_end: str = Form("11:15"),
    text_terms: str = Form("A2,routinem,Kontaktgespr,Austausch,Fragen"),
    ollama_model: str = Form("llama3.1:8b"),
    top_k: str = Form("5"),
    strong_group: str = Form("true"),
):
    topic_slug = topic

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM p_topics WHERE slug=%s", (topic_slug,))
            row = cur.fetchone()
            topic_title = row[0] if row else topic_slug  # fallback

            cur.execute("SELECT slug, title FROM p_topics ORDER BY title")
            topics = cur.fetchall()

    terms = [t.strip() for t in text_terms.split(",") if t.strip()]

    req = PlanUnitRequest(
        topic=topic_title,
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
        "topic": topic_slug,  # <-- slug fürs selected
        "level": level,
        "time_start": time_start,
        "time_end": time_end,
        "text_terms": text_terms,
        "ollama_model": ollama_model,
        "top_k": top_k,
        "strong_group": strong_group,
    }

    return templates.TemplateResponse(
        "preview.html",
        {"request": request, "form": form, "unit": unit, "topics": topics}
    )

@app.post("/unit", response_model=UnitResponse)
def create_unit(req: UnitCreateRequest):
    level_code = req.level.strip().upper()
    if level_code not in {"A1","A2","B1","B2","C1","C2"}:
        raise HTTPException(status_code=400, detail="level muss A1..C2 sein")

    topic_title = (req.topic or "").strip() or None
    topic_slug = (req.topic_slug or "").strip() or (slugify(topic_title) if topic_title else None)

    plan_json = json.dumps(req.plan, ensure_ascii=False)
    lang_json = json.dumps(req.language_support, ensure_ascii=False)

    with db() as conn:
        with conn.cursor() as cur:
            # level_id holen
            cur.execute("SELECT id FROM p_levels WHERE code=%s", (level_code,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail=f"Level {level_code} nicht in p_levels (seed_product.sql ausführen?)")
            level_id = row[0]

            # topic upsert (optional)
            topic_id = None
            if topic_title:
                if not topic_slug:
                    topic_slug = slugify(topic_title)
                cur.execute("SELECT id FROM p_topics WHERE slug=%s", (topic_slug,))
                trow = cur.fetchone()
                if trow:
                    topic_id = trow[0]
                else:
                    cur.execute(
                        "INSERT INTO p_topics (slug, title) VALUES (%s, %s) RETURNING id",
                        (topic_slug, topic_title)
                    )
                    topic_id = cur.fetchone()[0]

            # unit insert
            cur.execute(
                """
                INSERT INTO p_units (
                    level_id, topic_id, time_start, time_end, strong_group, title, notes, plan, language_support
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                RETURNING id, created_at, updated_at
                """,
                (level_id, topic_id, req.time_start, req.time_end, req.strong_group, req.title, req.notes, plan_json, lang_json)
            )
            unit_id, created_at, updated_at = cur.fetchone()

            # citations insert (optional)
            out_citations: List[UnitCitationOut] = []
            if req.citations:
                # optional: prüfen, ob chunk_id existiert
                for c in req.citations:
                    cur.execute(
                        """
                        INSERT INTO p_unit_citations (unit_id, chunk_id, score, quote)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (unit_id, c.chunk_id, float(c.score), c.quote or "")
                    )
                    cid = cur.fetchone()[0]
                    out_citations.append(UnitCitationOut(
                        id=str(cid),
                        chunk_id=c.chunk_id,
                        score=float(c.score),
                        quote=c.quote or ""
                    ))

            conn.commit()

    return UnitResponse(
        id=str(unit_id),
        created_at=created_at.isoformat(),
        updated_at=updated_at.isoformat(),
        level=level_code,
        topic=topic_title,
        time_start=req.time_start,
        time_end=req.time_end,
        strong_group=req.strong_group,
        title=req.title,
        notes=req.notes,
        plan=req.plan,
        language_support=req.language_support,
        citations=out_citations
    )
@app.get("/unit/{unit_id}", response_model=UnitResponse)
def get_unit(unit_id: str):
    # UUID parse
    try:
        uid = uuid.UUID(unit_id)
    except Exception:
        raise HTTPException(status_code=400, detail="unit_id ist kein gültiges UUID")

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.id, u.created_at, u.updated_at,
                       lv.code,
                       tp.title,
                       u.time_start, u.time_end, u.strong_group,
                       u.title, u.notes,
                       u.plan::text, u.language_support::text
                FROM p_units u
                JOIN p_levels lv ON lv.id = u.level_id
                LEFT JOIN p_topics tp ON tp.id = u.topic_id
                WHERE u.id = %s
                """,
                (str(uid),)
            )
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="Unit nicht gefunden")

            (uid2, created_at, updated_at, level_code, topic_title,
             time_start, time_end, strong_group, title, notes,
             plan_txt, lang_txt) = r

            plan = json.loads(plan_txt) if plan_txt else {}
            language_support = json.loads(lang_txt) if lang_txt else {}

            # citations + optional chunk metadata (source/page/chunk_index)
            cur.execute(
                """
                SELECT c.id, c.chunk_id, c.score, c.quote,
                       d.source, d.page, d.chunk_index
                FROM p_unit_citations c
                LEFT JOIN doc_chunks d ON d.id = c.chunk_id
                WHERE c.unit_id = %s
                ORDER BY c.score DESC, c.created_at ASC
                """,
                (str(uid),)
            )
            crows = cur.fetchall()

    citations = [
        UnitCitationOut(
            id=str(x[0]),
            chunk_id=int(x[1]),
            score=float(x[2]),
            quote=x[3] or "",
            source=x[4],
            page=x[5],
            chunk_index=x[6],
        )
        for x in crows
    ]

    return UnitResponse(
        id=str(uid2),
        created_at=created_at.isoformat(),
        updated_at=updated_at.isoformat(),
        level=level_code,
        topic=topic_title,
        time_start=time_start or "",
        time_end=time_end or "",
        strong_group=bool(strong_group),
        title=title or "",
        notes=notes or "",
        plan=plan,
        language_support=language_support,
        citations=citations
    )
@app.get("/unit/{unit_id}/preview", response_class=HTMLResponse)
def unit_preview(request: Request, unit_id: str):
    # nutzt deinen bestehenden JSON-Endpoint intern
    unit = get_unit(unit_id)  # UnitResponse-Objekt (pydantic)
    # in dict umwandeln für Jinja
    unit_dict = unit.model_dump() if hasattr(unit, "model_dump") else unit.dict()
    return templates.TemplateResponse(
        "unit_preview.html",
        {"request": request, "unit": unit_dict}
    )
@app.post("/unit_from_plan_unit", response_model=UnitFromPlanUnitResponse)
def unit_from_plan_unit(request: Request, req: UnitFromPlanUnitRequest):
    # 1) Plan erzeugen
    plan_req = PlanUnitRequest(
        topic=req.topic,
        level=req.level,
        time_start=req.time_start,
        time_end=req.time_end,
        strong_group=req.strong_group,
        top_k=req.top_k,
        text_terms=req.text_terms or ["A2","routinem","Kontaktgespr","Austausch","Fragen"],
        ollama_model=req.ollama_model,
    )
    unit_plan = plan_unit(plan_req)

    # 2) Title bestimmen
    title = req.title or unit_plan.unit_title

    # 3) p_units speichern + citations speichern
    plan_payload = {"phases": unit_plan.phases, "materials": unit_plan.materials}

    # language_support ist bereits JSON
    lang_payload = unit_plan.language_support

    citations_in: List[UnitCitationIn] = []
    # ger.citations kommt aus ask_hybrid: source/page/chunk_index/score
    # doc_chunk_id ist dort nicht drin → wir speichern quote+score
    # und versuchen chunk_id zu ermitteln (best effort)
    saved_citations = 0

    with db() as conn:
        # level_id holen
        with conn.cursor() as cur:
            level_code = req.level.strip().upper()
            cur.execute("SELECT id FROM p_levels WHERE code=%s", (level_code,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail=f"Level {level_code} nicht in p_levels")
            level_id = row[0]

            # topic upsert
            topic_title = req.topic.strip()
            topic_slug = slugify(topic_title)
            cur.execute("SELECT id FROM p_topics WHERE slug=%s", (topic_slug,))
            trow = cur.fetchone()
            if trow:
                topic_id = trow[0]
            else:
                cur.execute(
                    "INSERT INTO p_topics (slug, title) VALUES (%s, %s) RETURNING id",
                    (topic_slug, topic_title)
                )
                topic_id = cur.fetchone()[0]

            # unit insert
            plan_json = json.dumps(plan_payload, ensure_ascii=False)
            lang_json = json.dumps(lang_payload, ensure_ascii=False)

            cur.execute(
                """
                INSERT INTO p_units (
                    level_id, topic_id, time_start, time_end, strong_group,
                    title, notes, plan, language_support
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                RETURNING id, created_at, updated_at
                """,
                (level_id, topic_id, req.time_start, req.time_end, req.strong_group,
                 title, req.notes, plan_json, lang_json)
            )
            unit_id, created_at, updated_at = cur.fetchone()

            # citations speichern
            for c in unit_plan.ger.get("citations", []):
                quote = ""
                # wenn du später quote direkt aus hit willst -> hier erweitern
                # fürs erste: source/page/chunk_index als quote
                quote = f"{c.get('source')} S.{c.get('page')} (Chunk {c.get('chunk_index')})"

                # chunk id versuchen zu finden: über source+page+chunk_index
                chunk_id = None
                try:
                    cur.execute(
                        """
                        SELECT id
                        FROM doc_chunks
                        WHERE source=%s AND page=%s AND chunk_index=%s
                        LIMIT 1
                        """,
                        (c.get("source"), c.get("page"), c.get("chunk_index"))
                    )
                    r2 = cur.fetchone()
                    if r2:
                        chunk_id = int(r2[0])
                except Exception:
                    chunk_id = None

                if not chunk_id:
                    # ohne chunk_id kann FK failen -> skip
                    continue

                cur.execute(
                    """
                    INSERT INTO p_unit_citations (unit_id, chunk_id, score, quote)
                    VALUES (%s,%s,%s,%s)
                    """,
                    (unit_id, chunk_id, float(c.get("score", 0.0)), quote)
                )
                saved_citations += 1

            conn.commit()

    # 4) Unit laden (inkl. citations)
    u = get_unit(str(unit_id))

    preview_url = f"/unit/{u.id}/preview"
    
    return UnitFromPlanUnitResponse(
        unit=u,
        preview_url=preview_url,
        citations_saved=saved_citations
    )

@app.get("/topics/{slug}", response_model=TopicWithUnitsOut)
def get_topic(slug: str):
    sql = """
    SELECT
        t.id,
        t.slug,
        t.title,
        COUNT(u.id) AS unit_count
    FROM p_topics t
    LEFT JOIN p_units u ON u.topic_id = t.id
    WHERE t.slug = %s
    GROUP BY t.id, t.slug, t.title;
    """

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (slug,))
            r = cur.fetchone()

    if not r:
        raise HTTPException(status_code=404, detail="Topic nicht gefunden")

    return TopicWithUnitsOut(
        id=str(r[0]),
        slug=r[1],
        title=r[2],
        unit_count=r[3]
    )