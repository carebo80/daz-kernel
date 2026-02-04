# app/api/search.py
from fastapi import APIRouter, Query
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pgvector import Vector

from app.core.settings import MODEL_NAME, TOP_K_DEFAULT
from app.core.schemas import SearchHit
from app.repo.db import db

_embedder = SentenceTransformer(MODEL_NAME)
router = APIRouter(tags=["search"])

def search_chunks(q: str, top_k: int, source_like: Optional[str] = None) -> List[SearchHit]:
    q_emb = _embedder.encode([q], normalize_embeddings=True)[0].tolist()
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

    return [
        SearchHit(
            id=r[0], source=r[1], page=r[2], chunk_index=r[3],
            content=r[4], score=float(r[5])
        )
        for r in rows
    ]

@router.get("/search", response_model=List[SearchHit])
def search(q: str = Query(..., min_length=2),
           top_k: int = TOP_K_DEFAULT,
           source_like: Optional[str] = None):
    return search_chunks(q=q, top_k=top_k, source_like=source_like)