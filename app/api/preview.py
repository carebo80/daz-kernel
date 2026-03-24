# app/api/preview.py
from __future__ import annotations

import uuid, json, os
from typing import Optional, Any

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.settings import logger, OLLAMA_MODEL
from app.core.schemas import PlanUnitRequest
from app.repo.db import db
from app.services.planning import create_plan_unit, load_phase_model_schema
from app.phases import _distribute_minutes

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def load_tag_titles(tag_type: str, codes: list[str]) -> dict[str, str]:
    codes = [c for c in codes if isinstance(c, str) and c]
    if not codes:
        return {}
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT code, title
                FROM p_tags
                WHERE type=%s AND code = ANY(%s)
                """,
                (tag_type, codes),
            )
            rows = cur.fetchall()
    return {code: title for code, title in rows}


def parse_phases_json(pj: str) -> list[dict[str, Any]]:
    if not isinstance(pj, str) or not pj.strip():
        return []
    try:
        tmp = json.loads(pj)
        if isinstance(tmp, list):
            return [x for x in tmp if isinstance(x, dict)]
    except Exception:
        return []
    return []


def compute_phase_rows(phase_model_code: str, duration: int) -> list[dict[str, Any]]:
    phase_schema = load_phase_model_schema(phase_model_code)
    schema_phases = (phase_schema or {}).get("phases") or []

    titles: list[str] = []
    mins: list[int] = []
    weights: list[float] = []

    for p in schema_phases:
        if not isinstance(p, dict):
            continue
        titles.append(str(p.get("title") or p.get("phase") or "Phase"))
        try:
            mins.append(int(p.get("minutes", 0)))
        except Exception:
            mins.append(0)
        try:
            weights.append(float(p.get("weight", 1.0)))
        except Exception:
            weights.append(1.0)

    if not titles:
        return []

    dist = _distribute_minutes(duration, weights=weights, mins=mins) or [0] * len(titles)
    dist = [(int(d) // 5) * 5 for d in dist]  # 5er

    rows: list[dict[str, Any]] = []
    for i in range(len(titles)):
        rows.append(
            {
                "title": titles[i],
                "minutes": dist[i],
                "weight": weights[i],
                # Defaults, damit Template/JS immer Keys hat
                "subtopic_code": "",
                "grammar_code": "",
                "rag_codes": [],
                "subtopic_title": "",
                "grammar_title": "",
                "rag_titles": [],
            }
        )
    return rows


def fetch_preview_lists(topic_code: Optional[str] = None):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT code, title FROM p_tags
                WHERE type='topic'
                ORDER BY title
            """)
            topics = cur.fetchall()

            cur.execute("""
                SELECT code, title FROM p_phase_models
                WHERE is_active=true
                ORDER BY title
            """)
            phase_models = cur.fetchall()

            cur.execute("""SELECT code, title FROM p_tags WHERE type='level' ORDER BY code""")
            levels = cur.fetchall()

            cur.execute("""SELECT code, title FROM p_tags WHERE type='grammar' ORDER BY title""")
            grammars = cur.fetchall()

            cur.execute("""SELECT code, title FROM p_tags WHERE type='method' ORDER BY title""")
            methods = cur.fetchall()

            cur.execute("""SELECT code, title FROM p_tags WHERE type='rag_term' ORDER BY title""")
            rag_terms = cur.fetchall()

            subtopics = []
            if topic_code:
                cur.execute(
                    "SELECT id FROM p_tags WHERE type='topic' AND code=%s LIMIT 1",
                    (topic_code,),
                )
                trow = cur.fetchone()
                if trow:
                    cur.execute(
                        """
                        SELECT code, title
                        FROM p_tags
                        WHERE type='subtopic' AND parent_id=%s
                        ORDER BY title
                        """,
                        (trow[0],),
                    )
                    subtopics = cur.fetchall()

    return topics, phase_models, levels, grammars, methods, rag_terms, subtopics


def merge_ui_into_phase_rows(phase_rows: list[dict[str, Any]], ui_phases: list[dict[str, Any]]) -> None:
    """Mutiert phase_rows in-place."""
    if not ui_phases or len(ui_phases) != len(phase_rows):
        return

    # Codes sammeln für Titel-Mapping
    sub_codes: list[str] = []
    gr_codes: list[str] = []
    rag_codes: list[str] = []

    for u in ui_phases:
        sc = u.get("subtopic")
        gc = u.get("grammar")
        rgs = u.get("rag_terms")

        if isinstance(sc, str) and sc:
            sub_codes.append(sc)
        if isinstance(gc, str) and gc:
            gr_codes.append(gc)
        if isinstance(rgs, list):
            rag_codes += [x for x in rgs if isinstance(x, str) and x]

    sub_map = load_tag_titles("subtopic", sub_codes)
    gr_map = load_tag_titles("grammar", gr_codes)
    rag_map = load_tag_titles("rag_term", rag_codes)

    for i in range(len(phase_rows)):
        u = ui_phases[i]
        p = phase_rows[i]

        # minutes
        try:
            p["minutes"] = int(u.get("minutes") or p["minutes"] or 0)
        except Exception:
            pass

        # selected codes
        p["subtopic_code"] = u.get("subtopic") if isinstance(u.get("subtopic"), str) else ""
        p["grammar_code"] = u.get("grammar") if isinstance(u.get("grammar"), str) else ""
        p["rag_codes"] = u.get("rag_terms") if isinstance(u.get("rag_terms"), list) else []

        # titles
        sc = p["subtopic_code"]
        gc = p["grammar_code"]
        rc = p["rag_codes"]

        p["subtopic_title"] = sub_map.get(sc, "") if isinstance(sc, str) else ""
        p["grammar_title"] = gr_map.get(gc, "") if isinstance(gc, str) else ""
        p["rag_titles"] = [rag_map.get(x, x) for x in rc if isinstance(x, str)]


@router.get("/preview", response_class=HTMLResponse)
def preview_get(
    request: Request,
    topic: str = "bank",
    level: str = "A2",
    phase_model_code: str = "five_phases",
    duration_minutes: int = 150,
    top_k: int = 5,
):
    # UI Lists
    topics, phase_models, levels, grammars, methods, rag_terms, subtopics = fetch_preview_lists(topic)

    # Raster
    try:
        dur = int(duration_minutes)
    except Exception:
        dur = 150
    if dur <= 0:
        dur = 150

    phase_rows = compute_phase_rows(phase_model_code.strip().lower(), dur)

    form = {
        "phase_model_code": phase_model_code.strip().lower(),
        "topic": topic,
        "level": level,
        "duration_minutes": str(dur),
        "top_k": str(top_k),
        "text_terms": "",
        "phases_json": "",
        "notes": "",
        "subtopics": [],
        "grammar": [],
        "methods": [],
        "rag_terms": [],
        "rag_auto": True,
    }

    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "form": form,
            "unit": None,
            "topics": topics,
            "phase_models": phase_models,
            "phase_rows": phase_rows,
            "levels": levels,
            "grammars": grammars,
            "methods": methods,
            "rag_terms": rag_terms,
            "subtopics": subtopics,
        },
    )


@router.post("/preview", response_class=HTMLResponse)
def preview_post(
    request: Request,
    topic: str = Form(...),
    level: str = Form("A2"),
    duration_minutes: str = Form("150"),
    text_terms: str = Form(""),
    top_k: str = Form("5"),
    phase_model_code: str = Form("five_phases"),
    phases_json: str = Form(""),
    notes: str = Form(""),
):
    rid = uuid.uuid4().hex[:8]
    logger.info(f"[{rid}] POST /preview topic={topic} level={level} phase_model={phase_model_code} dur={duration_minutes}")
    logger.info(f"[{rid}] phases_json={phases_json[:200]}...")

    # 1) UI-Phasen genau einmal parsen
    ui_phases = parse_phases_json(phases_json)
    logger.info(f"[{rid}] ui_phases_len={len(ui_phases)} ui_phases_keys={(list(ui_phases[0].keys()) if ui_phases else [])}")

    # 2) UI-Listen laden
    topics, phase_models, levels, grammars, methods, rag_terms, subtopics = fetch_preview_lists(topic)

    # 3) Dauer normalisieren
    try:
        dur = int(duration_minutes) if duration_minutes else 150
    except Exception:
        dur = 150
    dur = max(15, (dur // 5) * 5)

    # 4) Builder-Raster erzeugen und UI-Werte mergen
    phase_rows = compute_phase_rows(phase_model_code.strip().lower(), dur)
    merge_ui_into_phase_rows(phase_rows, ui_phases)

    # 5) Topic-Titel für LLM holen
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title FROM p_tags WHERE type='topic' AND code=%s LIMIT 1",
                (topic,),
            )
            row = cur.fetchone()
    topic_title = row[0] if row else topic

    # 6) text_terms normalisieren
    terms = [t.strip() for t in (text_terms or "").split(",") if t.strip()]

    # 7) Request fürs Planning
    req = PlanUnitRequest(
        topic=topic_title,
        level=level,
        duration_minutes=dur,
        top_k=int(top_k),
        text_terms=terms,
        ollama_model=OLLAMA_MODEL,
        phase_model_code=phase_model_code.strip().lower(),
        phases_json=phases_json,
    )

    # 8) Unit generieren
    unit = create_plan_unit(req, rid=rid)
    logger.info(f"[{rid}] unit_phases_len={len(unit.phases) if isinstance(unit.phases, list) else -1}")

    # 9) Titel-Mapping für Anzeige
    sub_codes = [p.get("subtopic") for p in ui_phases if isinstance(p.get("subtopic"), str)]
    gr_codes = [p.get("grammar") for p in ui_phases if isinstance(p.get("grammar"), str)]

    rag_codes = []
    for p in ui_phases:
        xs = p.get("rag_terms") or []
        if isinstance(xs, list):
            rag_codes += [x for x in xs if isinstance(x, str)]

    sub_map = load_tag_titles("subtopic", sub_codes)
    gr_map = load_tag_titles("grammar", gr_codes)
    rag_map = load_tag_titles("rag_term", rag_codes)

    # 10) unit.phases mit hübschen Titles anreichern
    if ui_phases and isinstance(unit.phases, list) and len(ui_phases) == len(unit.phases):
        enriched = []
        for i, ph in enumerate(unit.phases):
            u = ui_phases[i]
            sc = u.get("subtopic") or ""
            gc = u.get("grammar") or ""
            rc = u.get("rag_terms") if isinstance(u.get("rag_terms"), list) else []

            ph["subtopic"] = sub_map.get(sc) or sc or ""
            ph["grammar"] = gr_map.get(gc) or gc or ""
            ph["rag_terms"] = [rag_map.get(x, x) for x in rc]

            try:
                ph["minutes"] = int(u.get("minutes") or ph.get("minutes") or 0)
            except Exception:
                pass

            enriched.append(ph)

        unit.phases = enriched
    else:
        logger.warning(
            f"[{rid}] enrich skipped: ui_len={len(ui_phases)} "
            f"unit_len={len(unit.phases) if isinstance(unit.phases, list) else None}"
        )

    # 11) Form-State zurückgeben
    form = {
        "phase_model_code": phase_model_code.strip().lower(),
        "topic": topic,
        "level": level,
        "duration_minutes": str(dur),
        "text_terms": text_terms,
        "top_k": top_k,
        "phases_json": phases_json,
        "notes": notes,
        "subtopics": [],
        "grammar": [],
        "methods": [],
        "rag_terms": [],
        "rag_auto": True,
    }

    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "form": form,
            "unit": unit,
            "topics": topics,
            "phase_models": phase_models,
            "phase_rows": phase_rows,
            "levels": levels,
            "grammars": grammars,
            "methods": methods,
            "rag_terms": rag_terms,
            "subtopics": subtopics,
        },
    )
