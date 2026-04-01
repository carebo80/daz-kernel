from __future__ import annotations

from psycopg2.extras import Json
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.repo.db import db
from app.activity_types import get_activity_redirect, get_activity_type_config

import qrcode
import random
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from app.bingo_config import (
    build_bingo_session_config,
    get_bingo_items,
    get_bingo_mode_options,
    normalize_grid_size,
    validate_bingo_items,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def _gen_join_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _create_unique_join_code(cur, tries: int = 10) -> str:
    for _ in range(tries):
        code = _gen_join_code()
        cur.execute("SELECT 1 FROM activity_sessions WHERE join_code=%s LIMIT 1", (code,))
        if not cur.fetchone():
            return code
    raise RuntimeError("Could not generate unique join code")

def has_bingo(marked: list[int], grid_size: int) -> bool:
    marked_set = set(marked)

    # Zeilen
    for row in range(grid_size):
        row_indices = {row * grid_size + col for col in range(grid_size)}
        if row_indices.issubset(marked_set):
            return True

    # Spalten
    for col in range(grid_size):
        col_indices = {row * grid_size + col for row in range(grid_size)}
        if col_indices.issubset(marked_set):
            return True

    # Diagonale links oben -> rechts unten
    diag1 = {i * grid_size + i for i in range(grid_size)}
    if diag1.issubset(marked_set):
        return True

    # Diagonale rechts oben -> links unten
    diag2 = {i * grid_size + (grid_size - 1 - i) for i in range(grid_size)}
    if diag2.issubset(marked_set):
        return True

    return False

def shuffled_copy(items: list[str]) -> list[str]:
    copied = list(items)
    random.shuffle(copied)
    return copied

@router.get("/")
def root():
    return RedirectResponse(url="/admin/activities", status_code=303)

@router.get("/admin/activities", response_class=HTMLResponse)
def admin_activities(request: Request):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, config, is_active, created_at
                FROM activity_templates
                WHERE is_active = true
                ORDER BY created_at DESC, title
                """
            )
            rows = cur.fetchall()

    templates_list: list[dict[str, Any]] = []
    for r in rows:
        templates_list.append(
            {
                "id": str(r[0]),
                "type": r[1],
                "title": r[2],
                "config": r[3] or {},
                "is_active": r[4],
                "created_at": r[5],
            }
        )

    return templates.TemplateResponse(
        "admin/activities/index.html",
        {
            "request": request,
            "templates_list": templates_list,
        },
    )

@router.post("/admin/activities/{template_id}/start")
def start_activity_session(template_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, config
                FROM activity_templates
                WHERE id=%s AND is_active=true
                LIMIT 1
                """,
                (template_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Activity template not found")

            join_code = _create_unique_join_code(cur)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=8)

            cur.execute(
                """
                INSERT INTO activity_sessions (template_id, join_code, status, config, created_at, expires_at)
                VALUES (%s, %s, %s, %s, now(), %s)
                RETURNING id
                """,
                (template_id, join_code, "waiting", Json(row[3] or {}), expires_at),
            )
            sess = cur.fetchone()
            if not sess:
                raise RuntimeError("Could not create session")

        conn.commit()

    return RedirectResponse(url=f"/admin/activities/session/{sess[0]}", status_code=303)


@router.get("/admin/activities/session/{session_id}", response_class=HTMLResponse)
def admin_activity_session(request: Request, session_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.join_code, s.status, s.config, s.created_at, s.expires_at,
                       t.id, t.type, t.title
                FROM activity_sessions s
                JOIN activity_templates t ON t.id = s.template_id
                WHERE s.id=%s
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")

            cur.execute(
                """
                SELECT id, display_name, joined_at, last_seen_at, state
                FROM participant_sessions
                WHERE session_id=%s
                ORDER BY joined_at ASC
                """,
                (session_id,),
            )
            participants_raw = cur.fetchall()

    session = {
        "id": str(row[0]),
        "join_code": row[1],
        "status": row[2],
        "config": row[3] or {},
        "created_at": row[4],
        "expires_at": row[5],
        "template_id": str(row[6]),
        "type": row[7],
        "title": row[8],
    }

    participants = [
        {
            "id": str(p[0]),
            "display_name": p[1],
            "joined_at": p[2].isoformat() if p[2] else None,
            "last_seen_at": p[3].isoformat() if p[3] else None,
            "state": p[4] or {},
            "visitenkarte": (p[4] or {}).get("visitenkarte", {}),
            "submitted": bool((p[4] or {}).get("submitted")),
            "bingo": (p[4] or {}).get("bingo", {}),
            "marked_count": len((((p[4] or {}).get("bingo", {}) or {}).get("marked", [])) or []),
            "has_bingo": bool((((p[4] or {}).get("bingo", {}) or {}).get("has_bingo"))),
        }
        for p in participants_raw
    ]

    activity_type_cfg = get_activity_type_config(session["type"])
    supports_print = activity_type_cfg.get("supports_print", False)

    return templates.TemplateResponse(
        "admin/activities/session.html",
        {
            "request": request,
            "session": session,
            "participants": participants,
            "join_url": f"/join/{session['join_code']}",
            "supports_print": supports_print,
        },
    )

@router.get("/join/{join_code}", response_class=HTMLResponse)
def join_activity_get(request: Request, join_code: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.join_code, s.status, s.config, s.expires_at,
                       t.type, t.title
                FROM activity_sessions s
                JOIN activity_templates t ON t.id = s.template_id
                WHERE s.join_code=%s
                LIMIT 1
                """,
                (join_code,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Join code not found")

    session = {
        "id": str(row[0]),
        "join_code": row[1],
        "status": row[2],
        "config": row[3] or {},
        "expires_at": row[4],
        "type": row[5],
        "title": row[6],
    }

    if session["status"] == "closed":
        return HTMLResponse("Diese Session ist beendet.", status_code=403)

    if session["expires_at"] and session["expires_at"] < datetime.now(timezone.utc):
        return HTMLResponse("Diese Session ist abgelaufen.", status_code=403)

    return templates.TemplateResponse(
        "activities/join.html",
        {
            "request": request,
            "session": session,
        },
    )

@router.post("/join/{join_code}", response_class=HTMLResponse)
def join_activity_post(
    request: Request,
    join_code: str,
    display_name: str = Form(""),
):
    display_name = (display_name or "").strip()

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.join_code, s.status, s.config, s.expires_at,
                       t.type, t.title
                FROM activity_sessions s
                JOIN activity_templates t ON t.id = s.template_id
                WHERE s.join_code=%s
                LIMIT 1
                """,
                (join_code,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Join code not found")

            if row[2] == "closed":
                return HTMLResponse("Diese Session ist beendet.", status_code=403)

            if row[4] and row[4] < datetime.now(timezone.utc):
                return HTMLResponse("Diese Session ist abgelaufen.", status_code=403)

            session = {
                "id": str(row[0]),
                "join_code": row[1],
                "status": row[2],
                "config": row[3] or {},
                "expires_at": row[4],
                "type": row[5],
                "title": row[6],
            }

            if not display_name:
                return templates.TemplateResponse(
                    "activities/join.html",
                    {
                        "request": request,
                        "session": session,
                        "error": "Bitte Namen oder Pseudonym eingeben.",
                    },
                    status_code=400,
                )

            session_id = str(row[0])
            activity_type = row[5]

            cur.execute(
                """
                INSERT INTO participant_sessions (session_id, display_name, state, joined_at, last_seen_at)
                VALUES (%s, %s, %s, now(), now())
                RETURNING id
                """,
                (session_id, display_name, Json({})),
            )
            p = cur.fetchone()

            if not p:
                raise RuntimeError("Could not create participant session")

        conn.commit()

    participant_id = str(p[0])

    redirect_url = get_activity_redirect(activity_type, participant_id)
    if redirect_url:
        return RedirectResponse(url=redirect_url, status_code=303)

    return HTMLResponse(f"Joined session {session_id} as {display_name} ({participant_id})")

@router.get("/activity/visitenkarten/{participant_id}", response_class=HTMLResponse)
def activity_visitenkarten(request: Request, participant_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ps.id,
                    ps.display_name,
                    ps.state,
                    s.id,
                    s.join_code,
                    s.status,
                    s.config,
                    t.title,
                    t.type,
                    t.config
                FROM participant_sessions ps
                JOIN activity_sessions s ON s.id = ps.session_id
                JOIN activity_templates t ON t.id = s.template_id
                WHERE ps.id=%s
                LIMIT 1
                """,
                (participant_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Participant session not found")

    participant = {
        "id": str(row[0]),
        "display_name": row[1],
        "state": row[2] or {},
        "session_id": str(row[3]),
        "join_code": row[4],
        "session_status": row[5],
        "session_config": row[6] or {},
        "template_title": row[7],
        "template_type": row[8],
        "template_config": row[9] or {},
    }

    fields = participant["template_config"].get("fields", [])

    return templates.TemplateResponse(
        "activities/visitenkarten/form.html",
        {
            "request": request,
            "participant": participant,
            "fields": fields,
        },
    )

@router.post("/activity/visitenkarten/{participant_id}", response_class=HTMLResponse)
def submit_visitenkarte(
    request: Request,
    participant_id: str,
    vorname: str = Form(...),
    nachname: str = Form(...),
    adresse: str = Form(...),
    plz: str = Form(""),
    ort: str = Form(""),
    telefon: str = Form(""),
    email: str = Form(""),
):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT state
                FROM participant_sessions
                WHERE id = %s
                """,
                (participant_id,)
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Teilnehmer nicht gefunden")

            state = row[0] or {}

            state["visitenkarte"] = {
                "vorname": vorname,
                "nachname": nachname,
                "adresse": adresse,
                "plz": plz,
                "ort": ort,
                "telefon": telefon,
                "email": email,
            }
            state["submitted"] = True

            cur.execute(
                """
                UPDATE participant_sessions
                SET state = %s
                WHERE id = %s
                """,
                (Json(state), participant_id)
            )

        conn.commit()

    return HTMLResponse("""
    <html>
      <body style="font-family:sans-serif;padding:2rem;">
        <h2>Danke!</h2>
        <p>Deine Visitenkarte wurde gespeichert.</p>
      </body>
    </html>
    """)

@router.get("/qr/{join_code}")
def qr_code(join_code: str, request: Request):
    url = f"{request.base_url}join/{join_code}"

    img = qrcode.make(url)
    buf = BytesIO()
    img.save(buf, format="PNG") # type: ignore
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@router.get("/admin/activities/session/{session_id}/participants")
def admin_activity_session_participants(session_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, display_name, joined_at, last_seen_at, state
                FROM participant_sessions
                WHERE session_id=%s
                ORDER BY joined_at ASC
                """,
                (session_id,),
            )
            participants_raw = cur.fetchall()

        participants = [
        {
            "id": str(p[0]),
            "display_name": p[1],
            "joined_at": p[2].isoformat() if p[2] else None,
            "last_seen_at": p[3].isoformat() if p[3] else None,
            "state": p[4] or {},
            "visitenkarte": (p[4] or {}).get("visitenkarte", {}),
            "submitted": bool((p[4] or {}).get("submitted")),
            "bingo": (p[4] or {}).get("bingo", {}),
            "marked_count": len(((p[4] or {}).get("bingo", {}) or {}).get("marked", []) or []),
            "has_bingo": bool((((p[4] or {}).get("bingo", {}) or {}).get("has_bingo"))),
        }
        for p in participants_raw
    ]

    return JSONResponse({"participants": participants})

@router.post("/admin/activities/session/{session_id}/participant/{participant_id}/delete")
def delete_participant(session_id: str, participant_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM participant_sessions
                WHERE id=%s AND session_id=%s
                """,
                (participant_id, session_id),
            )
        conn.commit()

    return RedirectResponse(
        url=f"/admin/activities/session/{session_id}",
        status_code=303,
    )

@router.post("/admin/activities/session/{session_id}/clear")
def clear_activity_session(session_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM participant_sessions
                WHERE session_id=%s
                """,
                (session_id,),
            )
        conn.commit()

    return RedirectResponse(
        url=f"/admin/activities/session/{session_id}",
        status_code=303,
    )

@router.get("/admin/activities/session/{session_id}/print", response_class=HTMLResponse)
def print_session_cards(request: Request, session_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, display_name, state
                FROM participant_sessions
                WHERE session_id = %s
                ORDER BY joined_at ASC
                """,
                (session_id,)
            )
            rows = cur.fetchall()

    session = {
        "id": session_id,
        "title": "Visitenkarten"
    }

    participants = []
    for row in rows:
        state = row[2] or {}
        vk = state.get("visitenkarte")

        participants.append({
            "id": str(row[0]),
            "display_name": row[1],
            "state": state,
            "visitenkarte": vk,
        })

    return templates.TemplateResponse(
        "admin/activities/print.html",
        {
            "request": request,
            "session": session,
            "participants": participants,
        }
    )

@router.post("/admin/activities/session/{session_id}/close")
def close_activity_session(session_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE activity_sessions
                SET status = 'closed'
                WHERE id = %s
                """,
                (session_id,)
            )
        conn.commit()

    return RedirectResponse(
        url=f"/admin/activities/session/{session_id}",
        status_code=303
    )
@router.get("/activity/bingo/{participant_id}", response_class=HTMLResponse)
def bingo_board(request: Request, participant_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ps.id, ps.display_name, ps.state,
                       s.id, s.status, s.config,
                       t.title, t.type
                FROM participant_sessions ps
                JOIN activity_sessions s ON s.id = ps.session_id
                JOIN activity_templates t ON t.id = s.template_id
                WHERE ps.id = %s
                LIMIT 1
                """,
                (participant_id,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Participant not found")

            participant = {
                "id": str(row[0]),
                "display_name": row[1],
                "state": row[2] or {},
            }
            session = {
                "id": str(row[3]),
                "status": row[4],
                "config": row[5] or {},
                "title": row[6],
                "type": row[7],
            }

            if session["status"] == "closed":
                return HTMLResponse("Diese Session ist beendet.", status_code=403)

            config = session["config"]
            grid_size = int(config.get("grid_size", 4))
            description = config.get("description", "")
            image = config.get("image", "default.jpg")

            bingo_state = participant["state"].get("bingo", {})
            board_items = bingo_state.get("board_items")

            # Falls noch kein individuelles Board existiert: erzeugen und speichern
            if not board_items:
                base_items = config.get("items", [])
                board_items = shuffled_copy(base_items)

                state = participant["state"]
                state.setdefault("bingo", {})
                state["bingo"]["board_items"] = board_items

                cur.execute(
                    """
                    UPDATE participant_sessions
                    SET state = %s, last_seen_at = now()
                    WHERE id = %s
                    """,
                    (Json(state), participant_id),
                )
                conn.commit()

                participant["state"] = state
                bingo_state = state.get("bingo", {})

            raw_marked = bingo_state.get("marked", [])
            marked: list[int] = []
            for value in raw_marked:
                try:
                    marked.append(int(value))
                except (TypeError, ValueError):
                    continue

            bingo_won = bool(bingo_state.get("has_bingo", False))

    return templates.TemplateResponse(
        "activities/bingo/board.html",
        {
            "request": request,
            "participant": participant,
            "session": session,
            "title": session["title"],
            "grid_size": grid_size,
            "items": board_items,
            "description": description,
            "image": image,
            "marked": marked,
            "bingo_won": bingo_won,
        },
    )

@router.post("/activity/bingo/{participant_id}", response_class=HTMLResponse)
def bingo_board_update(
    request: Request,
    participant_id: str,
    marked: list[str] | None = Form(None),
):
    marked = marked or []

    marked_indices: list[int] = []
    for value in marked:
        try:
            marked_indices.append(int(value))
        except ValueError:
            continue

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ps.id, ps.display_name, ps.state,
                       s.id, s.status, s.config,
                       t.title, t.type
                FROM participant_sessions ps
                JOIN activity_sessions s ON s.id = ps.session_id
                JOIN activity_templates t ON t.id = s.template_id
                WHERE ps.id = %s
                LIMIT 1
                """,
                (participant_id,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Participant not found")

            participant = {
                "id": str(row[0]),
                "display_name": row[1],
                "state": row[2] or {},
            }
            session = {
                "id": str(row[3]),
                "status": row[4],
                "config": row[5] or {},
                "title": row[6],
                "type": row[7],
            }

            if session["status"] == "closed":
                return HTMLResponse("Diese Session ist beendet.", status_code=403)

            state = participant["state"]
            state.setdefault("bingo", {})

            # Falls aus irgendeinem Grund noch kein Board existiert, jetzt erzeugen
            if not state["bingo"].get("board_items"):
                base_items = session["config"].get("items", [])
                state["bingo"]["board_items"] = shuffled_copy(base_items)

            grid_size = int(session["config"].get("grid_size", 4))
            bingo_won = has_bingo(marked_indices, grid_size)

            state["bingo"]["marked"] = marked_indices
            state["bingo"]["has_bingo"] = bingo_won

            cur.execute(
                """
                UPDATE participant_sessions
                SET state = %s, last_seen_at = now()
                WHERE id = %s
                """,
                (Json(state), participant_id),
            )

        conn.commit()

    if request.headers.get("HX-Request") == "true":
        bingo_state = state.get("bingo", {})
        raw_marked = bingo_state.get("marked", [])
        normalized_marked: list[int] = []
        for value in raw_marked:
            try:
                normalized_marked.append(int(value))
            except (TypeError, ValueError):
                continue

        return templates.TemplateResponse(
            "activities/bingo/board.html",
            {
                "request": request,
                "participant": participant,
                "session": session,
                "title": session["title"],
                "grid_size": grid_size,
                "items": bingo_state.get("board_items", []),
                "description": session["config"].get("description", ""),
                "image": session["config"].get("image", "default.jpg"),
                "marked": normalized_marked,
                "bingo_won": bingo_won,
            },
        )

    return RedirectResponse(
        url=f"/activity/bingo/{participant_id}",
        status_code=303,
    )

@router.get("/admin/activities/{template_id}/configure", response_class=HTMLResponse)
def configure_activity_template(request: Request, template_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, config, is_active
                FROM activity_templates
                WHERE id=%s AND is_active=true
                LIMIT 1
                """,
                (template_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Activity template not found")

    template = {
        "id": str(row[0]),
        "type": row[1],
        "title": row[2],
        "config": row[3] or {},
        "is_active": row[4],
    }

    if template["type"] != "bingo":
        return RedirectResponse(url=f"/admin/activities/{template_id}/start", status_code=303)

    return templates.TemplateResponse(
        "admin/activities/configure_bingo.html",
        {
            "request": request,
            "template": template,
            "mode_options": get_bingo_mode_options(),
            "selected_mode": "letters",
            "selected_grid_size": 4,
            "custom_items": "",
            "custom_description": "",
            "number_min": 0,
            "number_max": 20,
            "error": None,
        },
    )
@router.post("/admin/activities/{template_id}/start-configured")
def start_configured_activity_session(
    request: Request,
    template_id: str,
    activity_type: str = Form(""),
    mode: str = Form("letters"),
    grid_size: str = Form("4"),
    custom_items: str = Form(""),
    custom_description: str = Form(""),
    number_min: str = Form("0"),
    number_max: str = Form("20"),
):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, title, config
                FROM activity_templates
                WHERE id=%s AND is_active=true
                LIMIT 1
                """,
                (template_id,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Activity template not found")

            template = {
                "id": str(row[0]),
                "type": row[1],
                "title": row[2],
                "config": row[3] or {},
            }

            if template["type"] != "bingo":
                return RedirectResponse(url=f"/admin/activities/{template_id}/start", status_code=303)

            normalized_grid_size = normalize_grid_size(grid_size, default=4)
            try:
                normalized_number_min = int(number_min)
            except (TypeError, ValueError):
                normalized_number_min = 0

            try:
                normalized_number_max = int(number_max)
            except (TypeError, ValueError):
                normalized_number_max = 20
            try:
                items = get_bingo_items(
                    mode=mode,
                    grid_size=normalized_grid_size,
                    custom_items_text=custom_items,
                    number_min=normalized_number_min,
                    number_max=normalized_number_max,
                )
            except ValueError as exc:
                return templates.TemplateResponse(
                    "admin/activities/configure_bingo.html",
                    {
                        "request": request,
                        "template": template,
                        "mode_options": get_bingo_mode_options(),
                        "selected_mode": mode,
                        "selected_grid_size": normalized_grid_size,
                        "custom_items": custom_items,
                        "custom_description": custom_description,
                        "number_min": normalized_number_min,
                        "number_max": normalized_number_max,
                        "error": str(exc),
                    },
                    status_code=400,
                )
            errors = validate_bingo_items(items, normalized_grid_size)

            if errors:
                return templates.TemplateResponse(
                    "admin/activities/configure_bingo.html",
                    {
                        "request": request,
                        "template": template,
                        "mode_options": get_bingo_mode_options(),
                        "selected_mode": mode,
                        "selected_grid_size": normalized_grid_size,
                        "custom_items": custom_items,
                        "custom_description": custom_description,
                        "error": " ".join(errors),
                    },
                    status_code=400,
                )

            session_config = build_bingo_session_config(
                base_config=template["config"],
                mode=mode,
                grid_size=normalized_grid_size,
                items=items,
                custom_description=custom_description,
                number_min=normalized_number_min,
                number_max=normalized_number_max,
            )

            join_code = _create_unique_join_code(cur)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=8)

            cur.execute(
                """
                INSERT INTO activity_sessions (template_id, join_code, status, config, created_at, expires_at)
                VALUES (%s, %s, %s, %s, now(), %s)
                RETURNING id
                """,
                (template_id, join_code, "waiting", Json(session_config), expires_at),
            )
            sess = cur.fetchone()

            if not sess:
                raise RuntimeError("Could not create session")

        conn.commit()

    return RedirectResponse(url=f"/admin/activities/session/{sess[0]}", status_code=303)

@router.get("/admin/activities/session/{session_id}/play", response_class=HTMLResponse)
def admin_activity_play(request: Request, session_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.join_code, s.status, s.config, s.expires_at,
                       t.type, t.title
                FROM activity_sessions s
                JOIN activity_templates t ON t.id = s.template_id
                WHERE s.id=%s
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Session not found")

            session = {
                "id": str(row[0]),
                "join_code": row[1],
                "status": row[2],
                "config": row[3] or {},
                "expires_at": row[4],
                "type": row[5],
                "title": row[6],
            }

            cur.execute(
                """
                SELECT id, display_name, joined_at, last_seen_at, state
                FROM participant_sessions
                WHERE session_id=%s
                ORDER BY joined_at ASC
                """,
                (session_id,),
            )
            participants_raw = cur.fetchall()

    participants = [
        {
            "id": str(p[0]),
            "display_name": p[1],
            "joined_at": p[2].isoformat() if p[2] else None,
            "last_seen_at": p[3].isoformat() if p[3] else None,
            "state": p[4] or {},
            "marked_count": len((((p[4] or {}).get("bingo", {}) or {}).get("marked", [])) or []),
            "has_bingo": bool((((p[4] or {}).get("bingo", {}) or {}).get("has_bingo"))),
        }
        for p in participants_raw
    ]

    grid_size = int(session["config"].get("grid_size", 4))
    items = session["config"].get("items", [])
    raw_teacher_called_items = session["config"].get("teacher_called_items", [])
    teacher_called_items: list[int] = []

    teacher_last_called_index = session["config"].get("teacher_last_called_index")
    teacher_last_called_item = None

    if isinstance(teacher_last_called_index, int) and 0 <= teacher_last_called_index < len(items):
        teacher_last_called_item = items[teacher_last_called_index]

    for value in raw_teacher_called_items:
        try:
            teacher_called_items.append(int(value))
        except (TypeError, ValueError):
            continue

    return templates.TemplateResponse(
        "admin/activities/play.html",
        {
            "request": request,
            "session": session,
            "participants": participants,
            "grid_size": grid_size,
            "items": items,
            "teacher_called_items": teacher_called_items,
            "teacher_last_called_index": teacher_last_called_index,
            "teacher_last_called_item": teacher_last_called_item,
        },
    )

@router.post("/admin/activities/session/{session_id}/play", response_class=HTMLResponse)
def admin_activity_play_update(
    request: Request,
    session_id: str,
    teacher_called_items: list[str] | None = Form(None),
    action: str = Form(""),
):
    teacher_called_items = teacher_called_items or []

    called_indices: list[int] = []
    for value in teacher_called_items:
        try:
            called_indices.append(int(value))
        except ValueError:
            continue

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.join_code, s.status, s.config, s.expires_at,
                       t.type, t.title
                FROM activity_sessions s
                JOIN activity_templates t ON t.id = s.template_id
                WHERE s.id=%s
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Session not found")

            session = {
                "id": str(row[0]),
                "join_code": row[1],
                "status": row[2],
                "config": row[3] or {},
                "expires_at": row[4],
                "type": row[5],
                "title": row[6],
            }

            config = session["config"]
            items = config.get("items", [])

            previous_called = set((config.get("teacher_called_items", []) or []))
            current_called = set(called_indices)

            raw_teacher_last_called_index = session["config"].get("teacher_last_called_index")
            teacher_last_called_index = None

            try:
                if raw_teacher_last_called_index is not None:
                    teacher_last_called_index = int(raw_teacher_last_called_index)
            except (TypeError, ValueError):
                teacher_last_called_index = None

            # Zufälliges noch nicht markiertes Element hinzufügen
            if action == "random_pick":
                available_indices = [i for i in range(len(items)) if i not in current_called]
                if available_indices:
                    picked = random.choice(available_indices)
                    called_indices.append(picked)
                    called_indices = sorted(set(called_indices))
                    teacher_last_called_index = picked
            else:
                # Bei normalem Klicken versuchen wir das neu gesetzte Feld zu erkennen
                newly_added = [i for i in called_indices if i not in previous_called]
                if newly_added:
                    teacher_last_called_index = newly_added[-1]
                else:
                    # Falls nur abgewählt wurde und das letzte Item nicht mehr aktiv ist, leeren
                    if teacher_last_called_index not in called_indices:
                        teacher_last_called_index = None

            config["teacher_called_items"] = called_indices
            config["teacher_last_called_index"] = teacher_last_called_index

            cur.execute(
                """
                UPDATE activity_sessions
                SET config=%s
                WHERE id=%s
                """,
                (Json(config), session_id),
            )

        conn.commit()

    grid_size = int(config.get("grid_size", 4))
    items = config.get("items", [])

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(
            "admin/activities/play_board_partial.html",
            {
                "request": request,
                "session": session,
                "grid_size": grid_size,
                "items": items,
                "teacher_called_items": called_indices,
            },
        )

    return RedirectResponse(url=f"/admin/activities/session/{session_id}/play", status_code=303)

@router.get("/admin/activities/session/{session_id}/play/participants", response_class=HTMLResponse)
def admin_activity_play_participants(request: Request, session_id: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.join_code, s.status, s.config, s.expires_at,
                       t.type, t.title
                FROM activity_sessions s
                JOIN activity_templates t ON t.id = s.template_id
                WHERE s.id=%s
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Session not found")

            session = {
                "id": str(row[0]),
                "join_code": row[1],
                "status": row[2],
                "config": row[3] or {},
                "expires_at": row[4],
                "type": row[5],
                "title": row[6],
            }

            cur.execute(
                """
                SELECT id, display_name, joined_at, last_seen_at, state
                FROM participant_sessions
                WHERE session_id=%s
                ORDER BY joined_at ASC
                """,
                (session_id,),
            )
            participants_raw = cur.fetchall()

    participants = [
        {
            "id": str(p[0]),
            "display_name": p[1],
            "joined_at": p[2].isoformat() if p[2] else None,
            "last_seen_at": p[3].isoformat() if p[3] else None,
            "state": p[4] or {},
            "marked_count": len((((p[4] or {}).get("bingo", {}) or {}).get("marked", [])) or []),
            "has_bingo": bool((((p[4] or {}).get("bingo", {}) or {}).get("has_bingo"))),
        }
        for p in participants_raw
    ]

    return templates.TemplateResponse(
        "admin/activities/play_participants_partial.html",
        {
            "request": request,
            "session": session,
            "participants": participants,
        },
    )