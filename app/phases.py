from __future__ import annotations

from typing import Any, Dict, List, Optional

def _distribute_minutes(duration: int, weights: List[float], mins: Optional[List[int]] = None) -> List[int]:
    """
    Verteilt duration auf n Phasen.
    - Wenn mins angegeben (teilweise/komplett), werden diese bevorzugt.
    - Rest wird gemäss weights verteilt.
    """
    n = len(weights)
    if n == 0:
        return []

    # Start: vorgegebene Minuten übernehmen, wo vorhanden
    base = [0] * n
    if mins:
        for i in range(min(n, len(mins))):
            if isinstance(mins[i], int) and mins[i] >= 0:
                base[i] = mins[i]

    fixed = sum(base)
    remaining = max(duration - fixed, 0)

    # Wenn alles fix ist oder remaining=0: ggf. kürzen/skalieren, falls fixed > duration
    if remaining == 0:
        if fixed <= duration:
            return base
        # fixed zu gross -> proportional runter skalieren
        scale = duration / fixed if fixed else 0
        scaled = [int(round(x * scale)) for x in base]
        # Rundungsdiff korrigieren
        diff = duration - sum(scaled)
        for i in range(abs(diff)):
            idx = i % n
            scaled[idx] += 1 if diff > 0 else -1
        return scaled

    # weights normalisieren (falls alle 0 -> gleich verteilen)
    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * n
        wsum = float(n)

    raw = [remaining * (w / wsum) for w in weights]
    alloc = [int(x) for x in raw]
    # Rest-Minuten via grösste Nachkommaanteile verteilen
    diff = remaining - sum(alloc)
    frac = sorted([(raw[i] - alloc[i], i) for i in range(n)], reverse=True)
    for k in range(diff):
        alloc[frac[k % n][1]] += 1

    return [base[i] + alloc[i] for i in range(n)]


def build_phases(
    topic: str,
    level: str,
    duration: int,
    phase_schema: dict | None = None,
) -> List[Dict[str, Any]]:
    phase_schema = phase_schema or {}
    schema_phases = phase_schema.get("phases")

    if not isinstance(schema_phases, list) or not schema_phases:
        raise ValueError("Phase schema missing/invalid: expected phase_schema['phases'] as non-empty list")

    titles: List[str] = []
    aims: List[str] = []
    activities: List[str] = []
    mins: List[int] = []
    weights: List[float] = []

    for p in schema_phases:
        if not isinstance(p, dict):
            continue

        title = str(p.get("title") or p.get("phase") or "Phase")
        aim_tpl = str(p.get("aim_tpl") or p.get("aim") or "")
        act_tpl = str(p.get("activity_tpl") or p.get("activity") or "")

        aim = aim_tpl.format(topic=topic, level=level) if aim_tpl else ""
        activity = act_tpl.format(topic=topic, level=level) if act_tpl else ""

        titles.append(title)
        aims.append(aim)
        activities.append(activity)

        m = p.get("minutes")
        if isinstance(m, (int, float, str)):
            try:
                mins.append(int(m))
            except ValueError:
                mins.append(0)
        else:
            mins.append(0)

        w = p.get("weight")
        if isinstance(w, (int, float, str)):
            try:
                weights.append(float(w))
            except ValueError:
                weights.append(1.0)
        else:
            weights.append(1.0)

    if not titles:
        raise ValueError("Phase schema invalid: no valid phase dict entries")

    dist = _distribute_minutes(duration, weights=weights, mins=mins)

    out: List[Dict[str, Any]] = []
    for i in range(len(titles)):
        out.append({
            "phase": titles[i],
            "minutes": dist[i],
            "aim": aims[i] or f"Arbeiten an {topic}",
            "activity": activities[i] or "Übung/Arbeitsauftrag gemäss Schema.",
        })
    return out
