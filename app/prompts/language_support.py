# app/prompts/language_support.py
from __future__ import annotations
import json
from typing import Any, Dict, List

def build_language_support_prompt(
    *,
    topic: str,
    level: str,
    strong_group: bool,
    duration: int,
    context: str,
    group_desc: str,
    complexity_rules: str,
    ger_answer: str,
    ger_citations: List[Dict[str, Any]] | List[Any],
    debug: bool = False,
) -> str:
    # Optional: Zitate kompakt als Text (damit das LLM weniger ausrastet)
    citations_txt = ""
    try:
        citations_txt = json.dumps(ger_citations, ensure_ascii=False)[:2500]
    except Exception:
        citations_txt = str(ger_citations)[:2500]

    return f"""
Du bist ein professioneller DaZ/DaF-Unterrichtsplaner
mit didaktischer Ausbildung und Erfahrung in Erwachsenenbildung.

Aufgabe: Erzeuge Sprachmittel (Wortschatz, Redemittel, Grammatikfokus)
und genau 2 Mini-Dialoge passend zum Thema.

Parameter (verbindlich):
- Thema: {topic}
- Niveau: {level}
- strong_group: {strong_group}
- Dauer: {duration} Minuten
- Kontext: {context}
- Gruppe: {group_desc}

WICHTIG:
- Verwende ausschliesslich Inhalte, die zum Thema "{topic}" passen.
- Nutze idiomatisches Deutsch (keine wörtlichen Übersetzungen aus dem Englischen).
  Beispiel: statt "Ich bin gut, danke." -> "Mir geht’s gut, danke." / "Gut, danke."
- Verwende KEINE Standardszenarien (Bank/Hotel/Restaurant), ausser wenn "{topic}" es verlangt.

Didaktische Steuerung:
{complexity_rules}

GeR-/RAG-Basis (nutzen für Lernziel-Nähe, nicht kopieren):
- RAG-Answer: {ger_answer}
- RAG-Citations (kompakt): {citations_txt}

VERPFLICHTEND:
- Ausgabe MUSS genau 2 Mini-Dialoge enthalten.
- mini_dialogues darf NICHT leer sein.
- Dialoge müssen klar zum Thema "{topic}" passen.

Gib außerdem:
- 10–15 Wortschatz-Einträge
- 6–10 Redemittel
- 1–2 Grammatik-Foki passend zu {level}

Gib als EINZIGES Ergebnis gültiges JSON zurück (kein Markdown, keine Erklärungen).

JSON-Schema:
{{
  "vocabulary": [{{"word":"...","note":"kurz"}}],
  "phrases": [{{"de":"...","function":"..."}}],
  "grammar_focus": [{{"topic":"...","examples":["...","..."]}}],
  "mini_dialogues": [
    {{
      "title":"...",
      "lines":[
        {{"role":"...","text":"..."}},
        {{"role":"...","text":"..."}}
      ]
    }}
  ]
}}

{"DEBUG: Begründe kurz didaktische Entscheidungen (kurz, im Feld _debug) und gib _debug als String zurück." if debug else ""}
""".strip()
