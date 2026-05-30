import json
from pathlib import Path

from app.similarity_index import match_similarity


def test_similarity_matches_train_dialogues() -> None:
    train_path = Path(__file__).resolve().parent.parent / "app" / "data" / "train.json"
    records = json.loads(train_path.read_text(encoding="utf-8"))
    hits = 0
    for record in records:
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in record["messages"])
        expected_flags = record.get("expected_red_flags", [])
        expected = expected_flags[0]["category"] if expected_flags else "clean"
        got = match_similarity(dialogue)
        if got == expected:
            hits += 1
    assert hits >= len(records) * 0.9
