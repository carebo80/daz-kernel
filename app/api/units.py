from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
import json
import uuid

from app.repo.db import db
from app.utils import slugify
from app.core.schemas import (
    UnitCreateRequest,
    UnitResponse,
    UnitCitationOut,
    UnitCitationIn,   # optional, falls du es schon drin hast
)

router = APIRouter(prefix="/units", tags=["units"])
templates = Jinja2Templates(directory="templates")


@router.post("", response_model=UnitResponse)
def create_unit(req: UnitCreateRequest):
    level_code = req.level.strip().upper()
    if level_code not in {"A1", "A2", "B1", "B2", "C1", "C2"}:
        raise HTTPException(status_code=400, detail="level muss A1..C2 sein")

    topic_title = (req.topic or "").strip() or None
    topic_slug = (req.topic_slug or "").strip() or (slugify(topic_title) if topic_title else None)

    plan_json = json.dumps(req.plan or {}, ensure_ascii=False)
    lang_json = json.dumps(req.language_support or {}, ensure_ascii=False)

    out_citations: List[UnitCitationOut] = []

    with db() as conn:
        with conn.cursor() as cur:
            # level_id holen
            cur.execute("SELECT id FROM p_levels WHERE code=%s", (level_code,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=400,
                    detail=f"Level {level_code} nicht in p_levels (seed_product.sql ausführen?)",
                )
            level_id = row[0]

            # topic upsert (optional)
            topic_id = None
            if topic_title:
                if not topic_slug:
                    topic_slug = slugify(topic_title)

                cur.execute("SELECT id FROM p_topics WHERE slug=%s", (topic_slug,))
                trow = cur.fetchone()
                if trow:
                    topic_id = trow[0]
                else:
                    cur.execute(
                        "INSERT INTO p_topics (slug, title) VALUES (%s, %s) RETURNING id",
                        (topic_slug, topic_title),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise HTTPException(500, "DB-Fehler: topic nicht gefunden")
                    topic_id = row[0]

            # unit insert
            cur.execute(
                """
                INSERT INTO p_units (
                    level_id, topic_id, time_start, time_end, strong_group, title, notes, plan, language_support
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                RETURNING id, created_at, updated_at
                """,
                (
                    level_id,
                    topic_id,
                    req.time_start,
                    req.time_end,
                    req.strong_group,
                    req.title,
                    req.notes,
                    plan_json,
                    lang_json,
                ),
            )

            row = cur.fetchone()
            if not row:
                raise HTTPException(500, "DB-Fehler: topic nicht gefunden")
            unit_id, created_at, updated_at = row

            # citations insert (optional)
            if getattr(req, "citations", None):
                for c in req.citations:
                    cur.execute(
                        """
                        INSERT INTO p_unit_citations (unit_id, chunk_id, score, quote)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (unit_id, c.chunk_id, float(c.score), c.quote or ""),
                    )

                    row = cur.fetchone()
                    if not row:
                        raise HTTPException(500, "DB-Fehler: topic nicht gefunden")
                    cid = row[0]

                    out_citations.append(
                        UnitCitationOut(
                            id=str(cid),
                            chunk_id=c.chunk_id,
                            score=float(c.score),
                            quote=c.quote or "",
                        )
                    )

            conn.commit()

    return UnitResponse(
        id=str(unit_id),
        created_at=created_at.isoformat(),
        updated_at=updated_at.isoformat(),
        level=level_code,
        topic=topic_title,
        time_start=req.time_start,
        time_end=req.time_end,
        strong_group=req.strong_group,
        title=req.title,
        notes=req.notes,
        plan=req.plan or {},
        language_support=req.language_support or {},
        citations=out_citations,
    )


@router.get("/{unit_id}", response_model=UnitResponse)
def get_unit(unit_id: str):
    # UUID parse
    try:
        uid = uuid.UUID(unit_id)
    except Exception:
        raise HTTPException(status_code=400, detail="unit_id ist kein gültiges UUID")

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.id, u.created_at, u.updated_at,
                       lv.code,
                       tp.title,
                       u.time_start, u.time_end, u.strong_group,
                       u.title, u.notes,
                       u.plan::text, u.language_support::text
                FROM p_units u
                JOIN p_levels lv ON lv.id = u.level_id
                LEFT JOIN p_topics tp ON tp.id = u.topic_id
                WHERE u.id = %s
                """,
                (str(uid),),
            )
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="Unit nicht gefunden")

            (
                uid2,
                created_at,
                updated_at,
                level_code,
                topic_title,
                time_start,
                time_end,
                strong_group,
                title,
                notes,
                plan_txt,
                lang_txt,
            ) = r

            plan = json.loads(plan_txt) if plan_txt else {}
            language_support = json.loads(lang_txt) if lang_txt else {}

            # citations + chunk metadata
            cur.execute(
                """
                SELECT c.id, c.chunk_id, c.score, c.quote,
                       d.source, d.page, d.chunk_index
                FROM p_unit_citations c
                LEFT JOIN doc_chunks d ON d.id = c.chunk_id
                WHERE c.unit_id = %s
                ORDER BY c.score DESC, c.created_at ASC
                """,
                (str(uid),),
            )
            crows = cur.fetchall()

    citations = [
        UnitCitationOut(
            id=str(x[0]),
            chunk_id=int(x[1]),
            score=float(x[2]),
            quote=x[3] or "",
            source=x[4],
            page=x[5],
            chunk_index=x[6],
        )
        for x in crows
    ]

    return UnitResponse(
        id=str(uid2),
        created_at=created_at.isoformat(),
        updated_at=updated_at.isoformat(),
        level=level_code,
        topic=topic_title,
        time_start=time_start or "",
        time_end=time_end or "",
        strong_group=bool(strong_group),
        title=title or "",
        notes=notes or "",
        plan=plan,
        language_support=language_support,
        citations=citations,
    )


@router.get("/{unit_id}/preview", response_class=HTMLResponse)
def unit_preview(request: Request, unit_id: str):
    unit = get_unit(unit_id)  # UnitResponse (Pydantic)
    unit_dict = unit.model_dump() if hasattr(unit, "model_dump") else unit.dict()
    return templates.TemplateResponse(
        "unit_preview.html",
        {"request": request, "unit": unit_dict},
    )
