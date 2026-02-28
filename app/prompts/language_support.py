# app/prompts/language_support.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _json_compact(x: Any, limit: int = 2500) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)[:limit]
    except Exception:
        return str(x)[:limit]


def build_language_support_prompt(
    *,
    topic: str,
    level: str,
    duration: int,
    context: str,
    group_desc: str,
    ger_answer: str,
    ger_citations: List[Dict[str, Any]] | List[Any],
    phase_model_code: Optional[str] = None,
    phase_schema: Optional[Dict[str, Any]] = None,
    phase_grid_block: str = "",
    debug: bool = False,
) -> str:
    """
    Prompt für LLM: generiert Sprachmittel + genau 2 Mini-Dialoge als JSON.

    phase_model_code / phase_schema:
      - optional, um ein DB-gesteuertes Phasenmodell als Kontext reinzugeben
      - KEINE Pflicht, damit bestehende Calls nicht brechen
    """

    citations_txt = _json_compact(ger_citations, limit=2500)
    phase_schema_txt = _json_compact(phase_schema or {}, limit=2500)

    # Harte Sprachregeln gegen typische “eingedeutschte” Fehler
    anti_calque_rules = f"""
SPRACHQUALITÄT (sehr wichtig):
- Schreibe idiomatisches, natürliches Deutsch für die Schweiz/Deutschland.
- KEINE wörtlichen Übersetzungen aus dem Englischen.
- Artikel/Kasus korrekt, besonders bei Nomen mit Artikel:
  - korrekt: "eine Kreditkarte", nicht: "einen Kreditkarte"
- Höfliche Bitten korrekt:
  - korrekt: "Kann ich bitte Wasser haben?" / "Könnte ich bitte Wasser bekommen?"
  - NICHT: "Darf ich bitte Wasser?"
- Verben passend zur Situation:
  - "eine Kreditkarte beantragen" / "eine Kreditkarte bestellen" (nicht "kaufen")
  - "Ich trinke gerne Kaffee." (nicht "Ich esse gerne Kaffee.")
- Keine inhaltlichen Brüche: wenn Thema "{topic}" ist, dann keine Restaurant-/Hotel-Dialoge.
""".strip()

    # Optional: Phasenmodell als Kontext (DB-gesteuert)
    phase_block = ""
    if phase_model_code or (phase_schema and phase_schema != {}):
        phase_block = f"""
PHASENMODELL (für Konsistenz, nicht zwingend für Output-Struktur):
- phase_model_code: {phase_model_code or ""}
- phase_schema (kompakt): {phase_schema_txt}
Hinweis: Wenn das Phasenmodell Phasen/Schwerpunkte vorgibt, sollen Wortschatz/Redemittel/Grammatik
und die Mini-Dialoge dazu passen (z.B. mehr "Transfer"-Dialog, wenn Transferphase betont wird).
""".strip()
    
    phase_grid_txt = ""
    if phase_grid_block:
        phase_grid_txt = f"""
    {phase_grid_block}

    WICHTIG (hart):
    - Verwende EXAKT diese Phasen in gleicher Reihenfolge.
    - Verwende EXAKT die Minuten.
    - Gib phase_plan mit genau diesen Phasen zurück.
    """.strip()
    # Optional: Debug-Anweisung
    debug_block = (
        'DEBUG: Begründe kurz didaktische Entscheidungen im Feld "_debug" (String, max. 500 Zeichen).'
        if debug
        else ""
    )

    # Ein “Output-Contract”, damit das Modell weniger aus dem Schema ausbricht
    output_contract = f"""
AUSGABE-CONTRACT (hart):
- Wenn PHASENRASTER vorhanden ist, MUSS phase_plan enthalten sein.
- phase_plan MUSS die gleiche Länge haben wie das Raster.
- Wenn du unsicher bist, setze aim/activity leer, aber gib phase_plan trotzdem aus.
- Gib als EINZIGES Ergebnis gültiges JSON zurück (kein Markdown, keine Erklärungen).
- Felder müssen exakt so heissen: vocabulary, phrases, grammar_focus, mini_dialogues, phase_plan{", _debug" if debug else ""}.
- mini_dialogues: genau 2 Einträge.
- phase_plan: Array mit genau den Phasen aus dem PHASENRASTER (falls vorhanden), gleiche Reihenfolge & Minuten.
- Jedes phase_plan Element: phase, minutes, aim, activity.
- Jeder Dialog: 6–10 Zeilen, Rollen logisch, Sätze kurz & A2-gerecht.
- Jede Zeile muss natürliches Deutsch sein (keine komischen Kalke).
""".strip()

    return f"""
Du bist ein professioneller DaZ/DaF-Unterrichtsplaner
mit didaktischer Ausbildung und Erfahrung in Erwachsenenbildung.

AUFGABE:
Erzeuge Sprachmittel (Wortschatz, Redemittel, Grammatikfokus)
und genau 2 Mini-Dialoge passend zum Thema.

PARAMETER (verbindlich):
- Thema: {topic}
- Niveau: {level}
- Dauer: {duration} Minuten
- Kontext: {context}
- Gruppe: {group_desc}

DIDAKTISCHE STEUERUNG:

{anti_calque_rules}

{phase_block}

{phase_grid_txt}
GeR-/RAG-Basis (nutzen für Lernziel-Nähe, nicht kopieren):
- RAG-Answer: {ger_answer}
- RAG-Citations (kompakt): {citations_txt}

{output_contract}

Gib außerdem:
- 10–15 Wortschatz-Einträge (alltagsnah, thematisch)
- 6–10 Redemittel (funktional: fragen, klären, bitten, reagieren)
- 1–2 Grammatik-Foki passend zu {level} (mit 2–3 Beispielen)
WICHTIG: Halte dich exakt an das JSON-Schema. Gib keine zusätzlichen Felder aus.
Wenn phase_plan fehlt, gib die gesamte Antwort erneut korrekt aus.
JSON-Schema:
{{
  "vocabulary": [{{"word":"...","note":"kurz"}}],
  "phrases": [{{"de":"...","function":"..."}}],
  "grammar_focus": [{{"topic":"...","examples":["...","..."]}}],
  "phase_plan": [
    {{"phase":"...","minutes":10,"aim":"...","activity":"..."}}
  ],
  "mini_dialogues": [
    {{
      "title":"...",
      "lines":[
        {{"role":"...","text":"..."}},
        {{"role":"...","text":"..."}}
      ]
    }}
  ]{',' if debug else ''}
  {"\"_debug\": \"...\"" if debug else ""}
}}

{debug_block}
""".strip()
