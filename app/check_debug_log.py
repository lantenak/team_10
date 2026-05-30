"""Opt-in JSONL logging входящих /check для отладки на своём сервере."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.signals import compute_signal_scores
from app.train_registry import lookup_verbatim_train

_ENABLED = os.getenv("DEBUG_LOG_CHECKS", "").lower() in ("1", "true", "yes")
_LOG_PATH = Path(os.getenv("DEBUG_LOG_PATH", "logs/check_requests.jsonl"))
_RAW_SESSION = os.getenv("DEBUG_LOG_RAW_SESSION_ID", "").lower() in ("1", "true", "yes")
_LOCK = threading.Lock()


def _session_ref(session_id: str) -> str:
    if _RAW_SESSION:
        return session_id
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log_check_request(
    *,
    session_id: str,
    dialogue_text: str,
    predicted_category: str | None,
    processing_time_ms: int,
) -> None:
    """Append one JSON line. No-op unless DEBUG_LOG_CHECKS=true."""
    if not _ENABLED:
        return

    signals = compute_signal_scores(dialogue_text)
    top = signals.top_two()
    verbatim = lookup_verbatim_train(dialogue_text, session_id=session_id)

    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "session_ref": _session_ref(session_id),
        "n_lines": dialogue_text.count("\n") + 1 if dialogue_text else 0,
        "predicted_category": predicted_category,
        "processing_time_ms": processing_time_ms,
        "verbatim_train_hit": verbatim is not False,
        "verbatim_label": None if verbatim is False else verbatim,
        "signals_rule_hit": signals.rule_hit,
        "signal_top": top[0][0] if top else None,
        "signal_top_score": round(top[0][1], 2) if top else 0.0,
        "clean_boost": round(signals.clean_boost, 2),
        "dialogue": dialogue_text,
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        _ensure_parent(_LOG_PATH)
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)


def is_debug_logging_enabled() -> bool:
    return _ENABLED
