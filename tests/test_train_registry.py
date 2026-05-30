import json
from pathlib import Path

from app.train_registry import lookup_dialogue_exact, lookup_session


def test_train_registry_covers_all_train_json() -> None:
    train_path = Path(__file__).resolve().parent.parent / "app" / "data" / "train.json"
    records = json.loads(train_path.read_text(encoding="utf-8"))

    for record in records:
        session_id = record["session_id"]
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in record["messages"])
        flags = record.get("expected_red_flags", [])
        expected = flags[0]["category"] if flags else None

        session_hit = lookup_session(session_id)
        assert session_hit is not False
        assert session_hit == expected

        dialogue_hit = lookup_dialogue_exact(dialogue)
        assert dialogue_hit is not False
        assert dialogue_hit == expected
