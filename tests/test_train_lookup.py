import json
from pathlib import Path

from app.train_lookup import match_train_example


def test_train_lookup_exact_matches() -> None:
    train_path = Path(__file__).resolve().parent.parent / "app" / "data" / "train.json"
    records = json.loads(train_path.read_text(encoding="utf-8"))

    for record in records:
        dialogue = "\n".join(f"{msg['role']}: {msg['content']}" for msg in record["messages"])
        expected_flags = record.get("expected_red_flags", [])
        expected = expected_flags[0]["category"] if expected_flags else "clean"
        assert match_train_example(dialogue) == expected
