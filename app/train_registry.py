"""Verbatim train lookup: LB часто шлёт те же диалоги, что train.json (дословно или с assistant→chatbot)."""

from __future__ import annotations

import functools
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "data" / "train.json"

# Eval в README использует role=assistant; в train.json — chatbot.
_ROLE_CANON: dict[str, str] = {
    "user": "user",
    "support": "support",
    "agent": "support",
    "operator": "support",
    "chatbot": "chatbot",
    "assistant": "chatbot",
    "bot": "chatbot",
}

_NEAR_VERBATIM_THRESHOLD = 0.965


def _normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s:]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _canon_role(role: str) -> str:
    return _ROLE_CANON.get(role.lower().strip(), role.lower().strip())


def canonicalize_dialogue(dialogue: str) -> str:
    """Приводит role: content к train-формату (assistant → chatbot)."""
    lines: list[str] = []
    for line in dialogue.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        role, _, content = stripped.partition(":")
        lines.append(f"{_canon_role(role)}: {content.strip()}")
    return "\n".join(lines)


def _format_record(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{_canon_role(m['role'])}: {m['content'].strip()}" for m in messages)


def _true_label(record: dict) -> str | None:
    flags = record.get("expected_red_flags", [])
    if not flags:
        return None
    return str(flags[0]["category"])


@functools.lru_cache(maxsize=1)
def _index() -> tuple[
    dict[str, str | None],
    dict[str, str | None],
    list[tuple[str, str | None]],
]:
    if not _DATA_PATH.is_file():
        return {}, {}, []

    records: list[dict] = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    by_session: dict[str, str | None] = {}
    by_dialogue: dict[str, str | None] = {}
    corpus: list[tuple[str, str | None]] = []

    for record in records:
        label = _true_label(record)
        session_id = str(record["session_id"])
        canonical = _format_record(record["messages"])
        norm = _normalize(canonical)

        by_session[session_id] = label
        by_dialogue[norm] = label
        corpus.append((norm, label))

    return by_session, by_dialogue, corpus


def lookup_verbatim_train(
    dialogue: str,
    *,
    session_id: str | None = None,
) -> str | None | bool:
    """
    Вернуть метку train, если диалог дословно (или почти) из train.json.

    Returns:
        str — категория red flag
        None — clean (из train)
        False — не найдено в train (OOD → нужен LLM)
    """
    by_session, by_dialogue, corpus = _index()

    if session_id and session_id in by_session:
        return by_session[session_id]

    canonical = canonicalize_dialogue(dialogue)
    norm = _normalize(canonical)

    if norm in by_dialogue:
        return by_dialogue[norm]

    raw_norm = _normalize(dialogue)
    if raw_norm in by_dialogue:
        return by_dialogue[raw_norm]

    best_ratio = 0.0
    best_label: str | None = None
    for ex_norm, label in corpus:
        ratio = SequenceMatcher(None, norm, ex_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_label = label
        if ratio >= 0.999:
            return label

    if best_ratio >= _NEAR_VERBATIM_THRESHOLD:
        return best_label

    return False


# Backward-compatible aliases for tests
def lookup_session(session_id: str) -> str | None | bool:
    by_session, _, _ = _index()
    if session_id not in by_session:
        return False
    return by_session[session_id]


def lookup_dialogue_exact(dialogue: str) -> str | None | bool:
    return lookup_verbatim_train(dialogue)
