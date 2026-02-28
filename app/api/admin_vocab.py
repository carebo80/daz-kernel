from __future__ import annotations
import uuid
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.repo.db import db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def parse_csv_codes(s: str) -> List[str]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(x)
    # de-dupe, preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def fetch_tag_options() -> Dict[str, List[tuple[str, str]]]:
    """Lädt Optionen für Dropdown/Picker."""
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT code, title FROM p_tags WHERE type='subtopic' ORDER BY title")
            subtopics = cur.fetchall()
            cur.execute("SELECT code, title FROM p_tags WHERE type='grammar' ORDER BY title")
            grammars = cur.fetchall()
            cur.execute("SELECT code, title FROM p_tags WHERE type='rag_term' ORDER BY title")
            rag_terms = cur.fetchall()
    return {"subtopics": subtopics, "grammars": grammars, "rag_terms": rag_terms}


@router.get("/vocab/new", response_class=HTMLResponse)
def vocab_new(request: Request):
    opts = fetch_tag_options()
    form = {
        "lemma": "",
        "pos": "",
        "article": "",
        "plural": "",
        "level": "",
        "definition": "",
        "example": "",
        "tags_subtopic": "",
        "tags_grammar": "",
        "tags_rag_term": "",
    }
    return templates.TemplateResponse(
        "admin_vocab_new.html",
        {"request": request, "form": form, **opts},
    )


@router.post("/vocab/save")
def vocab_save(
    request: Request,
    vocab_id: Optional[str] = Form(None),
    lemma: str = Form(...),
    pos: str = Form(""),
    article: str = Form(""),
    plural: str = Form(""),
    level: str = Form(""),
    definition: str = Form(""),
    example: str = Form(""),
    tags_subtopic: str = Form(""),
    tags_grammar: str = Form(""),
    tags_rag_term: str = Form(""),
):
    # 1) upsert vocab
    with db() as conn:
        with conn.cursor() as cur:
            if vocab_id:
                cur.execute(
                    """
                    UPDATE p_vocabulary
                    SET lemma=%s, pos=%s, article=%s, plural=%s, level=%s,
                        definition=%s, example=%s, updated_at=now()
                    WHERE id=%s
                    RETURNING id
                    """,
                    (lemma, pos, article, plural, level, definition, example, vocab_id),
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("Update failed: id not found")
                vid = str(row[0])
            else:
                cur.execute(
                    """
                    INSERT INTO p_vocabulary (lemma,pos,article,plural,level,definition,example)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                    """,
                    (lemma, pos, article, plural, level, definition, example),
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("Insert failed: no id returned")
                vid = str(row[0])

            # 2) tags ersetzen (pro type)
            # wir nutzen p_vocabulary_tags + p_tags (type, code) UNIQUE
            def replace_tags(tag_type: str, csv_codes: str):
                codes = parse_csv_codes(csv_codes)
                # alle existierenden dieses types löschen
                cur.execute(
                    """
                    DELETE FROM p_vocabulary_tags vt
                    USING p_tags t
                    WHERE vt.vocabulary_id=%s
                      AND vt.tag_id=t.id
                      AND t.type=%s
                    """,
                    (vid, tag_type),
                )
                if not codes:
                    return
                # neue setzen (nur tags die es gibt)
                cur.execute(
                    """
                    INSERT INTO p_vocabulary_tags (vocabulary_id, tag_id)
                    SELECT %s, t.id
                    FROM p_tags t
                    WHERE t.type=%s AND t.code = ANY(%s)
                    ON CONFLICT DO NOTHING
                    """,
                    (vid, tag_type, codes),
                )

            replace_tags("subtopic", tags_subtopic)
            replace_tags("grammar", tags_grammar)
            replace_tags("rag_term", tags_rag_term)

        conn.commit()

    # Redirect zurück zum Formular (oder später Edit-Seite)
    return RedirectResponse(url="/admin/vocab/new", status_code=303)