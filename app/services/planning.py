# app/services/planning.py
from __future__ import annotations

from typing import Any, Dict
from datetime import date
import time
import json
from app.core.settings import logger
from app.core.schemas import PlanUnitRequest, PlanUnitResponse, AskHybridRequest
from app.repo.db import db
from app.services.rag import ask_hybrid
from app.services.ollama import ollama_generate
from app.utils import parse_json_loose

from app.phases import build_phases
from app.prompts.language_support import build_language_support_prompt

def load_phase_model_schema(code: str) -> Dict[str, Any]:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT schema
                FROM p_phase_models
                WHERE code=%s AND is_active=true
                LIMIT 1
                """,
                (code,),
            )
            row = cur.fetchone()
    return row[0] if row else {}

def _complexity_rules(level: str) -> str:
    lvl = (level or "").upper()
    return (
        f"- Niveau {lvl}: kurze, klare Sätze und Muster.\n"
        "- Fokus auf Verständlichkeit, typische Fehler (Artikel/Kasus/Verbposition).\n"
        "- Output lieber korrekt als kreativ."
    )

def create_plan_unit(req: PlanUnitRequest, rid: str | None = None) -> PlanUnitResponse:
    rid = rid or "-"

    topic = req.topic.strip()
    level = req.level.strip().upper()
    duration = int(req.duration_minutes)
    phase_model_code = req.phase_model_code.strip().lower()

    # 1) DB-Phasenmodell laden
    t0 = time.perf_counter()
    phase_schema = load_phase_model_schema(phase_model_code)
    logger.info(f"[{rid}] phase_schema_load_ms={int((time.perf_counter()-t0)*1000)} phase_model={phase_model_code}")

    logger.info(f"[{rid}] Creating plan: topic={topic}, level={level}, duration={duration}, phase_model={phase_model_code}")

    # 2) Phasen bauen (einmal!)
    t0 = time.perf_counter()
    phases = build_phases(
        topic=topic,
        level=level,
        duration=duration,
        phase_schema=phase_schema,
    )
    grid_titles: list[str] = []
    grid_minutes: list[int] = []
    phase_grid_block = ""
    logger.info(f"[{rid}] build_phases_ms={int((time.perf_counter()-t0)*1000)} phases={len(phases)}")
    
    ui_phases: list[dict[str, Any]] | None = None
    pj = getattr(req, "phases_json", None)

    if isinstance(pj, str) and pj.strip():
        try:
            tmp = json.loads(pj)
            if isinstance(tmp, list):
                ui_phases = [x for x in tmp if isinstance(x, dict)]
        except Exception:
            ui_phases = None

    if ui_phases:
        lines = []
        for i, p in enumerate(ui_phases, start=1):
            title = str(p.get("title") or f"Phase {i}")
            try:
                minutes = int(p.get("minutes", 0))
            except Exception:
                minutes = 0

            grid_titles.append(title)
            grid_minutes.append(minutes)
            lines.append(f"{i}. {title} – {minutes} Min.")

        if lines:
            phase_grid_block = "PHASENRASTER (vom Plan Builder, verbindlich):\n" + "\n".join(lines)


    # 2) Kontext/Gruppe/Regeln (damit Prompt nicht rot ist)
    context = getattr(req, "context", None) or "Erwachsenenbildung, DaZ, Schweiz"
    group_desc = getattr(req, "group_desc", None) or "A2-Kurs, gemischte Vorkenntnisse, praxisnah"

    # 3) GeR / RAG holen
    hybrid_req = AskHybridRequest(
        question=f"GeR/CEFR-Bezug für Lernziele im Thema {topic}",
        level=level,
        top_k=req.top_k,
        text_terms=req.text_terms,
    )
    ger_resp = ask_hybrid(hybrid_req)

    # 5) Language support Prompt + LLM call
    ls_prompt = build_language_support_prompt(
        topic=topic,
        level=level,
        duration=duration,
        context=context,
        group_desc=group_desc,
        ger_answer=ger_resp.answer,
        ger_citations=ger_resp.citations,
        phase_model_code=phase_model_code,
        phase_schema=phase_schema,
        phase_grid_block=phase_grid_block,
    )

    ls_raw = ollama_generate(req.ollama_model, ls_prompt)
    try:
        language_support = parse_json_loose(ls_raw)
    except Exception:
        language_support = {"raw": ls_raw}
    
    if isinstance(language_support, dict):
        logger.info(f"[{rid}] language_support_keys={list(language_support.keys())}")
    else:
        logger.warning(f"[{rid}] language_support_not_dict")

    logger.info(f"[{rid}] Ollama model used: {req.ollama_model}")
    logger.info(f"[{rid}] phase_grid_block_present={bool(phase_grid_block)} grid_len={len(grid_titles)}")
    # --- phase_plan aus AI übernehmen, aber ans Raster binden ---
    ai_phase_plan = None
    if isinstance(language_support, dict):
        pp = language_support.get("phase_plan")
        if isinstance(pp, list):
            ai_phase_plan = [x for x in pp if isinstance(x, dict)]

    def _norm_title(s: str) -> str:
        return " ".join((s or "").strip().lower().split())

    if ai_phase_plan and grid_titles:
        # wir erwarten gleiche Anzahl + gleiche titles/minutes
        if len(ai_phase_plan) == len(grid_titles):
            ok = True
            out = []
            for i, item in enumerate(ai_phase_plan):
                phase_name = str(item.get("phase") or "")
                try:
                    minutes = int(item.get("minutes", -999))
                except Exception:
                    minutes = -999

                if _norm_title(phase_name) != _norm_title(grid_titles[i]):
                    ok = False
                    break
                if minutes != grid_minutes[i]:
                    ok = False
                    break

                out.append({
                    "phase": grid_titles[i],
                    "minutes": grid_minutes[i],
                    "subtopic": ui_phases[i].get("subtopic", "") if ui_phases else "",
                    "grammar": ui_phases[i].get("grammar", "") if ui_phases else "",
                    "rag_terms": ui_phases[i].get("rag_terms", []) if ui_phases else [],
                    "aim": str(item.get("aim") or ""),
                    "activity": str(item.get("activity") or ""),
                })

            if ok:
                phases = out
            else:
                logger.warning(f"[{rid}] phase_plan rejected (mismatch vs grid). using fallback phases.")
        else:
            logger.warning(f"[{rid}] phase_plan rejected (len mismatch). using fallback phases.")

    unit_title = f"{level} – {topic}: Unterrichtseinheit ({duration} Min.)"

    meta = {
        "topic": topic,
        "level": level,
        "date": date.today().isoformat(),
        "duration_minutes": duration,
        "ollama_model": req.ollama_model,
        "phase_model_code": phase_model_code,
    }

    ger: Dict[str, Any] = {"answer": ger_resp.answer, "citations": ger_resp.citations}

    return PlanUnitResponse(
        unit_title=unit_title,
        meta=meta,
        ger=ger,
        language_support=language_support,
        phases=phases,
        materials=[],
    )
