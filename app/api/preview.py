# app/api/preview.py

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.repo.db import db
from app.core.schemas import PlanUnitRequest
from app.services.planning import create_plan_unit
from app.core.settings import OLLAMA_MODEL

router = APIRouter()

templates = Jinja2Templates(directory="templates")


# -----------------------------
# GET: Formular anzeigen
# -----------------------------
@router.get("/preview", response_class=HTMLResponse)
def preview_get(request: Request):

    # Topics laden
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT slug, title FROM p_topics ORDER BY title")
            topics = cur.fetchall()

    # Default-Form
    form = {
        "topic": "bank",
        "level": "A2",
        "time_start": "08:45",
        "time_end": "11:15",
        "text_terms": "A2,routinem,Kontaktgespr,Austausch,Fragen",
        "ollama_model": OLLAMA_MODEL,
        "top_k": "5",
        "strong_group": "true",
    }

    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "form": form,
            "unit": None,
            "topics": topics,
        }
    )


# -----------------------------
# POST: Formular absenden
# -----------------------------
@router.post("/preview", response_class=HTMLResponse)
def preview_post(
    request: Request,

    topic: str = Form(...),
    level: str = Form("A2"),
    time_start: str = Form("08:45"),
    time_end: str = Form("11:15"),
    text_terms: str = Form(""),
    ollama_model: str = Form("llama3.1:8b"),
    top_k: str = Form("5"),
    strong_group: str = Form("true"),
):

    # Topic-Titel holen
    with db() as conn:
        with conn.cursor() as cur:

            cur.execute(
                "SELECT title FROM p_topics WHERE slug=%s",
                (topic,)
            )
            row = cur.fetchone()

            topic_title = row[0] if row else topic

            cur.execute("SELECT slug, title FROM p_topics ORDER BY title")
            topics = cur.fetchall()

    # Text-Terms aufsplitten
    terms = [t.strip() for t in text_terms.split(",") if t.strip()]

    # Request bauen
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

    # Plan erzeugen
    unit = create_plan_unit(req)

    # Formular wieder füllen
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

    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "form": form,
            "unit": unit,
            "topics": topics,
        }
    )
