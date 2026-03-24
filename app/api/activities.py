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

import qrcode
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse

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
            "joined_at": p[2],
            "last_seen_at": p[3],
            "state": p[4] or {},
        }
        for p in participants_raw
    ]

    return templates.TemplateResponse(
        "admin/activities/session.html",
        {
            "request": request,
            "session": session,
            "participants": participants,
            "join_url": f"/join/{session['join_code']}",
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

    if activity_type == "visitenkarten":
        return RedirectResponse(
            url=f"/activity/visitenkarten/{participant_id}",
            status_code=303,
        )

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
    img.save(buf, format="PNG")
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