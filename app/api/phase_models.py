from fastapi import APIRouter
from typing import List, Dict, Any
from app.repo.db import db

router = APIRouter(tags=["phase_models"])

@router.get("/phase_models")
def list_phase_models() -> List[Dict[str, Any]]:
    sql = """
    SELECT id::text, code, title, description
    FROM p_phase_models
    WHERE is_active = true
    ORDER BY title;
    """
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return [
        {"id": r[0], "code": r[1], "title": r[2], "description": r[3] or ""}
        for r in rows
    ]
