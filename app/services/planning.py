# app/services/planning.py
from __future__ import annotations
from datetime import date
from typing import Any, Dict, List
from app.core.schemas import PlanUnitRequest, PlanUnitResponse, AskHybridRequest
from app.phases import build_phases
from app.prompts.language_support import build_language_support_prompt
from app.core.schemas import AskHybridRequest
from app.services.ollama import ollama_generate
from app.services.rag import ask_hybrid
from app.utils import parse_json_loose, minutes_between

def create_plan_unit(req: PlanUnitRequest) -> PlanUnitResponse:
    topic = req.topic.strip()
    level = req.level.strip().upper()
    strong_group = bool(req.strong_group)
    phase_model = req.phase_model

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
        text_terms=req.text_terms or [level, topic, "Kontaktgespr", "Austausch", "Fragen"],
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
    language_support: Dict[str, Any] = {
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
