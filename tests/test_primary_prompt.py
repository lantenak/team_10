from app.prompts import build_primary_classification_prompt, select_relevant_few_shots
from app.signals import compute_signal_scores


def test_select_relevant_few_shots_bounded() -> None:
    shots = select_relevant_few_shots(focus_categories=["information_extraction"], max_total=12)
    assert 4 <= len(shots) <= 12
    assert any(label == "information_extraction" for label, _ in shots)
    assert any(label == "clean" for label, _ in shots)


def test_primary_prompt_smaller_than_full() -> None:
    from app.prompts import build_classification_prompt

    dialogue = "user: test\nsupport: ok"
    signals = compute_signal_scores(dialogue)
    primary = build_primary_classification_prompt(dialogue, focus_categories=["scope_violation"])
    full = build_classification_prompt(dialogue)
    assert len(primary) < len(full)
