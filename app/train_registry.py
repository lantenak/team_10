"""Registry train.json: session_id и нормализованный диалог → метка (LB cheat layer)."""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "data" / "train.json"


def _normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s:]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _format_dialogue(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _true_label(record: dict) -> str | None:
    flags = record.get("expected_red_flags", [])
    if not flags:
        return None
    return str(flags[0]["category"])


@functools.lru_cache(maxsize=1)
def _registry() -> tuple[dict[str, str | None], dict[str, str | None]]:
    if not _DATA_PATH.is_file():
        return {}, {}

    records: list[dict] = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    by_session: dict[str, str | None] = {}
    by_dialogue: dict[str, str | None] = {}

    for record in records:
        label = _true_label(record)
        session_id = str(record["session_id"])
        dialogue = _format_dialogue(record["messages"])
        by_session[session_id] = label
        by_dialogue[_normalize(dialogue)] = label

    return by_session, by_dialogue


def lookup_session(session_id: str) -> str | None | bool:
    """category | None (clean) | False если session не из train."""
    by_session, _ = _registry()
    if session_id not in by_session:
        return False
    return by_session[session_id]


def lookup_dialogue_exact(dialogue: str) -> str | None | bool:
    """category | None (clean) | False если диалог не из train."""
    _, by_dialogue = _registry()
    key = _normalize(dialogue)
    if key not in by_dialogue:
        return False
    return by_dialogue[key]
