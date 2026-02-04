from typing import List, Dict, Any
from app.core.schemas import AskHybridRequest, AskResponse
from app.repo.db import db
from app.api.search import search_chunks  # <-- sauberer Service Import

def ask_hybrid(req: AskHybridRequest) -> AskResponse:
    q = req.question
    top_k = req.top_k
    lvl = req.level or "A1"

    terms = req.text_terms or [lvl, "routinem", "Kontaktgespr", "Austausch", "Fragen"]

    # 1) Text-Treffer
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
                text_hits.extend(cur.fetchall())

    # dedup
    seen = set()
    dedup_text_hits = []
    for r in text_hits:
        if r[0] in seen:
            continue
        seen.add(r[0])
        dedup_text_hits.append(r)

    # 2) Vector-Treffer (jetzt korrekt IN der Funktion)
    vector_hits = []
    try:
        vector_hits = search_chunks(q=q, top_k=top_k)
    except Exception:
        vector_hits = []

    # 3) Antwort bauen (dein bestehender Code)
    citations = []
    parts = []

    if dedup_text_hits:
        parts.append("Texttreffer (exakte Stellen):")
        for (id_, source, page, chunk_index, content) in dedup_text_hits[:top_k]:
            snippet = (content[:280] + "…") if len(content) > 280 else content
            parts.append(f"- {source}, S. {page} (Chunk {chunk_index})\n  » {snippet}")
            citations.append({"source": source, "page": page, "chunk_index": chunk_index, "score": 1.0})

    if vector_hits:
        parts.append("\nSemantische Treffer (Kontext/Operationalisierung):")
        for h in vector_hits[:top_k]:
            snippet = (h.content[:280] + "…") if len(h.content) > 280 else h.content
            parts.append(f"- {h.source}, S. {h.page} (Score {h.score:.3f})\n  » {snippet}")
            citations.append({"source": h.source, "page": h.page, "chunk_index": h.chunk_index, "score": h.score})

    if not parts:
        return AskResponse(answer="Keine Treffer gefunden.", citations=[])

    answer = f"Frage: {q}\n\n" + "\n\n".join(parts)
    return AskResponse(answer=answer, citations=citations)
