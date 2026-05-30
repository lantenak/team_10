from app.prompts import build_primary_classification_prompt, select_relevant_few_shots


def test_select_relevant_few_shots_bounded() -> None:
    shots = select_relevant_few_shots(focus_categories=["information_extraction"], max_total=12)
    assert 4 <= len(shots) <= 12
    assert any(label == "information_extraction" for label, _ in shots)
    assert any(label == "clean" for label, _ in shots)


def test_primary_prompt_uses_v5() -> None:
    dialogue = "user: test\nsupport: ok"
    primary = build_primary_classification_prompt(dialogue, focus_categories=["scope_violation"])
    assert "PROMPT_V5 — INTENT-BASED RED FLAG CLASSIFIER" in primary
    assert "FINAL DECISION RULE" in primary
    assert "ДИАЛОГ ДЛЯ КЛАССИФИКАЦИИ" in primary


def test_primary_and_full_prompt_same_base() -> None:
    from app.prompts import build_classification_prompt

    dialogue = "user: test\nsupport: ok"
    primary = build_primary_classification_prompt(dialogue, focus_categories=["scope_violation"])
    full = build_classification_prompt(dialogue)
    assert primary == full
