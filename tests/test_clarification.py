from app.clarification import DISAMBIGUATION_PAIRS, INTENT_PROBE_QUESTIONS
from app.prompts import build_intent_clarification_prompt


def test_intent_probe_catalog() -> None:
    assert len(INTENT_PROBE_QUESTIONS) >= 7
    labels = {label for _, _, label in INTENT_PROBE_QUESTIONS}
    assert "information_extraction" in labels
    assert "clean" in labels


def test_clarification_prompt_includes_probes() -> None:
    dialogue = "user: test\nsupport: ok"
    prompt = build_intent_clarification_prompt(
        dialogue,
        candidates=["information_extraction", "clean"],
    )
    assert "other_client_data" in prompt
    assert "information_extraction vs identity_deception" in prompt.lower() or "identity_deception" in prompt
    assert "user: test" in prompt
