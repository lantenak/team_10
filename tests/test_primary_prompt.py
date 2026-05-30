from app.gold_fewshots import GOLD_FEW_SHOT_EXAMPLES
from app.prompts import build_primary_classification_prompt, select_relevant_few_shots


def test_select_relevant_few_shots_bounded() -> None:
    shots = select_relevant_few_shots(focus_categories=["information_extraction"], max_total=12)
    assert 4 <= len(shots) <= 12
    assert any(label == "information_extraction" for label, _ in shots)
    assert any(label == "clean" for label, _ in shots)


def test_primary_prompt_uses_v5_and_gold() -> None:
    dialogue = "user: test\nsupport: ok"
    primary = build_primary_classification_prompt(dialogue, focus_categories=["scope_violation"])
    assert "PROMPT_V5 — INTENT-BASED RED FLAG CLASSIFIER" in primary
    assert "GOLD DIALOGUES (72 эталона)" in primary
    assert "FINAL DECISION RULE" in primary
    assert primary.count("\n---\n") >= 72
    assert "ДИАЛОГ ДЛЯ КЛАССИФИКАЦИИ" in primary


def test_gold_examples_count() -> None:
    assert len(GOLD_FEW_SHOT_EXAMPLES) == 72
