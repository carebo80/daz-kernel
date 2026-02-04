# app/api/topics.py

from fastapi import APIRouter, HTTPException
from typing import List, Optional

from app.repo.db import db
from app.core.schemas import TopicOut, TopicWithUnitsOut

router = APIRouter()


@router.get("/topics", response_model=List[TopicOut])
def list_topics(q: Optional[str] = None, limit: int = 200):

    sql = """
    SELECT id, slug, title
    FROM p_topics
    WHERE (%s IS NULL OR title ILIKE %s OR slug ILIKE %s)
    ORDER BY title ASC
    LIMIT %s;
    """

    with db() as conn:
        with conn.cursor() as cur:
            like = f"%{q}%" if q else None
            cur.execute(sql, (q, like, like, limit))
            rows = cur.fetchall()

    return [
        TopicOut(id=str(r[0]), slug=r[1], title=r[2])
        for r in rows
    ]


@router.get("/topics/{slug}", response_model=TopicWithUnitsOut)
def get_topic(slug: str):

    sql = """
    SELECT
        t.id,
        t.slug,
        t.title,
        COUNT(u.id) AS unit_count
    FROM p_topics t
    LEFT JOIN p_units u ON u.topic_id = t.id
    WHERE t.slug = %s
    GROUP BY t.id, t.slug, t.title;
    """

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (slug,))
            r = cur.fetchone()

    if not r:
        from fastapi import HTTPException
        raise HTTPException(404, "Topic nicht gefunden")

    return TopicWithUnitsOut(
        id=str(r[0]),
        slug=r[1],
        title=r[2],
        unit_count=r[3]
    )