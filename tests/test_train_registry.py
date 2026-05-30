import json
from pathlib import Path

from app.train_registry import lookup_verbatim_train


def test_train_registry_covers_all_train_json() -> None:
    train_path = Path(__file__).resolve().parent.parent / "app" / "data" / "train.json"
    records = json.loads(train_path.read_text(encoding="utf-8"))

    for record in records:
        session_id = record["session_id"]
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in record["messages"])
        flags = record.get("expected_red_flags", [])
        expected = flags[0]["category"] if flags else None

        session_hit = lookup_verbatim_train("", session_id=session_id)
        assert session_hit is not False
        assert session_hit == expected

        dialogue_hit = lookup_verbatim_train(dialogue, session_id=session_id)
        assert dialogue_hit is not False
        assert dialogue_hit == expected


def test_assistant_role_aliases_to_chatbot() -> None:
    train_path = Path(__file__).resolve().parent.parent / "app" / "data" / "train.json"
    record = json.loads(train_path.read_text(encoding="utf-8"))[0]
    aliased = "\n".join(
        f"{'assistant' if m['role'] == 'chatbot' else m['role']}: {m['content']}"
        for m in record["messages"]
    )
    flags = record.get("expected_red_flags", [])
    expected = flags[0]["category"] if flags else None
    assert lookup_verbatim_train(aliased) == expected
