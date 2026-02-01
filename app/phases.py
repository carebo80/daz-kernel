from __future__ import annotations
from typing import Any, Dict, List

def build_phases(
    *,
    topic: str,
    level: str,
    strong_group: bool,
    duration: int,
    model: str = "rita",
) -> List[Dict[str, Any]]:
    model = (model or "rita").lower()

    if model == "ariva":
        return [
            {"phase": "A – Anfangen/Aktivieren", "minutes": 10,
             "aim": f"Einstieg ins Thema {topic}",
             "activity": "Impulsfrage + kurzer Austausch."},
            {"phase": "R – Reaktivieren", "minutes": 10,
             "aim": "Vorwissen aktivieren",
             "activity": "Wortschatz sammeln + Satzmuster."},
            {"phase": "I – Informieren", "minutes": 20,
             "aim": "Neue Redemittel/Strukturen",
             "activity": "Mini-Dialoge + Fokusstellen."},
            {"phase": "V – Verarbeiten", "minutes": 30,
             "aim": "Üben/Anwenden",
             "activity": "Lückendialog → Partnerdialog."},
            {"phase": "A – Abschliessen", "minutes": 10,
             "aim": "Sichern + Mini-Check",
             "activity": "Exit-Ticket: 2 Sätze."},
        ]

    # default: RITA
    return [
        {"phase": "R – Ressourcen aktivieren", "minutes": 15,
         "aim": f"Vorwissen zu {topic} aktivieren",
         "activity": "Brainstorm + Wortschatz + Mini-Fragen."},
        {"phase": "I – Informationen verarbeiten", "minutes": 35,
         "aim": "Sprachmittel erarbeiten/üben",
         "activity": "Mini-Dialoge analysieren + Muster üben."},
        {"phase": "T – Transfer anbahnen", "minutes": 35,
         "aim": "Handlungsaufgabe / Rollenspiel",
         "activity": "Rollenspiel mit Variation je nach Gruppe."},
        {"phase": "A – Auswerten", "minutes": 15,
         "aim": "Feedback, Korrektur, Sicherung",
         "activity": "2 gute Sätze + 1 typischer Fehler."},
    ]
