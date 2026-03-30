from __future__ import annotations

from typing import Any


def visitenkarten_redirect(participant_id: str) -> str:
    return f"/activity/visitenkarten/{participant_id}"

def bingo_redirect(participant_id: str) -> str:
    return f"/activity/bingo/{participant_id}"

ACTIVITY_TYPES: dict[str, dict[str, Any]] = {
    "visitenkarten": {
        "label": "Visitenkarten",
        "state_key": "visitenkarte",
        "participant_template": "activities/visitenkarten/form.html",
        "print_template": "admin/activities/print.html",
        "supports_print": True,
        "participant_redirect": visitenkarten_redirect,
    },
    "bingo": {
        "label": "Bingo",
        "state_key": "bingo",
        "participant_template": "activities/bingo/board.html",
        "supports_print": False,
        "participant_redirect": bingo_redirect,
    },
}


def get_activity_type_config(activity_type: str) -> dict[str, Any]:
    return ACTIVITY_TYPES.get(activity_type, {})


def get_activity_state_key(activity_type: str, default: str = "form_data") -> str:
    cfg = get_activity_type_config(activity_type)
    return cfg.get("state_key", default)


def get_activity_redirect(activity_type: str, participant_id: str) -> str | None:
    cfg = get_activity_type_config(activity_type)
    redirect_fn = cfg.get("participant_redirect")
    if not redirect_fn:
        return None
    return redirect_fn(participant_id)