import json
from pathlib import Path

from app.check_debug_log import log_check_request


def test_debug_log_disabled_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEBUG_LOG_CHECKS", raising=False)
    log_path = tmp_path / "out.jsonl"
    monkeypatch.setenv("DEBUG_LOG_PATH", str(log_path))

    # Reload module flags — call with env unset
    import app.check_debug_log as mod

    monkeypatch.setattr(mod, "_ENABLED", False)
    log_check_request(
        session_id="session_test",
        dialogue_text="user: hi",
        predicted_category=None,
        processing_time_ms=10,
    )
    assert not log_path.exists()


def test_debug_log_writes_jsonl(tmp_path, monkeypatch) -> None:
    import app.check_debug_log as mod

    log_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(mod, "_ENABLED", True)
    monkeypatch.setattr(mod, "_LOG_PATH", log_path)
    monkeypatch.setattr(mod, "_RAW_SESSION", False)

    log_check_request(
        session_id="session_abc",
        dialogue_text="user: test",
        predicted_category="information_extraction",
        processing_time_ms=42,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["predicted_category"] == "information_extraction"
    assert row["session_ref"].startswith("sha256:")
    assert "session_abc" not in row["session_ref"]
    assert row["dialogue"] == "user: test"
