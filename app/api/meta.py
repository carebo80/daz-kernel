# app/api/meta.py
from fastapi import APIRouter, HTTPException
from app.repo.db import db
from app.core.settings import MODEL_NAME

router = APIRouter()

@router.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@router.get("/stats")
def stats():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM doc_chunks;")
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(500, "DB-Fehler: topic nicht gefunden")
            n = row[0]

            cur.execute("""
                SELECT source, COUNT(*)
                FROM doc_chunks
                GROUP BY source
                ORDER BY 2 DESC
            """)
            sources = [
                {"source": r[0], "count": r[1]}
                for r in cur.fetchall()
            ]

    return {
        "count": n,
        "sources": sources
    }