from __future__ import annotations

from typing import Any
import random


BINGO_MODE_LABELS: dict[str, str] = {
    "letters": "Buchstaben",
    "numbers": "Zahlen",
    "custom": "Freie Liste",
    "find_someone": "Finde jemanden, der …",
}


def get_bingo_mode_options() -> list[dict[str, str]]:
    return [{"value": key, "label": label} for key, label in BINGO_MODE_LABELS.items()]


def normalize_grid_size(value: str | int | None, default: int = 4) -> int:
    try:
        grid_size = int(value) # type: ignore
    except (TypeError, ValueError):
        return default

    if grid_size not in (4, 5):
        return default

    return grid_size


def required_item_count(grid_size: int) -> int:
    return grid_size * grid_size


def generate_letter_items(grid_size: int) -> list[str]:
    count = required_item_count(grid_size)
    start = ord("A")
    return [chr(start + i) for i in range(count)]


def generate_number_items(grid_size: int, number_min: int = 0, number_max: int = 20) -> list[str]:
    count = required_item_count(grid_size)

    if number_max < number_min:
        raise ValueError("number_max darf nicht kleiner als number_min sein.")

    pool = list(range(number_min, number_max + 1))

    if len(pool) < count:
        raise ValueError(
            f"Für ein {grid_size}x{grid_size}-Bingo werden mindestens {count} Zahlen benötigt, "
            f"aber der Bereich {number_min}–{number_max} enthält nur {len(pool)}."
        )

    selected = random.sample(pool, count)
    return [str(n) for n in selected]


def parse_custom_items(raw_text: str | None) -> list[str]:
    if not raw_text:
        return []

    lines = [line.strip() for line in raw_text.splitlines()]
    items = [line for line in lines if line]
    return items


def generate_find_someone_items(grid_size: int) -> list[str]:
    presets = [
        "hat Geschwister",
        "wohnt in einer Wohnung",
        "trinkt gern Kaffee",
        "spricht zwei Sprachen",
        "hat ein Hobby",
        "steht früh auf",
        "fährt mit dem Bus",
        "kocht gern",
        "liest gern",
        "hat Kinder",
        "mag Musik",
        "macht Sport",
        "wohnt in der Stadt",
        "hat ein Haustier",
        "isst gern Obst",
        "schreibt gern",
        "hat heute Zeit",
        "lernt Deutsch zu Hause",
        "kommt aus Europa",
        "arbeitet am Wochenende",
        "hat ein Velo",
        "mag Tee",
        "schaut Serien",
        "geht gern spazieren",
        "spricht Englisch",
    ]
    return presets[: required_item_count(grid_size)]


def get_bingo_items(
    mode: str,
    grid_size: int,
    custom_items_text: str | None = None,
    number_min: int = 0,
    number_max: int = 20,
) -> list[str]:
    if mode == "letters":
        return generate_letter_items(grid_size)

    if mode == "numbers":
        return generate_number_items(grid_size, number_min=number_min, number_max=number_max)

    if mode == "find_someone":
        return generate_find_someone_items(grid_size)

    if mode == "custom":
        return parse_custom_items(custom_items_text)

    return []


def validate_bingo_items(items: list[str], grid_size: int) -> list[str]:
    errors: list[str] = []
    needed = required_item_count(grid_size)

    if len(items) != needed:
        errors.append(
            f"Für ein {grid_size}x{grid_size}-Bingo werden genau {needed} Einträge benötigt. Aktuell: {len(items)}."
        )

    return errors


def build_bingo_session_config(
    base_config: dict[str, Any] | None,
    mode: str,
    grid_size: int,
    items: list[str],
    custom_description: str | None = None,
    number_min: int | None = None,
    number_max: int | None = None,
) -> dict[str, Any]:
    config = dict(base_config or {})

    config["grid_size"] = grid_size
    config["mode"] = mode
    config["items"] = items

    if number_min is not None:
        config["number_min"] = number_min

    if number_max is not None:
        config["number_max"] = number_max

    if custom_description is not None and custom_description.strip():
        config["description"] = custom_description.strip()

    return config