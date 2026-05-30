from app.prompts import TRAJECTORY_CLASSIFIER_BODY, build_primary_classification_prompt, select_relevant_few_shots
from app.trajectory_prompt import CALIBRATION_LIBRARY


def test_select_relevant_few_shots_bounded() -> None:
    shots = select_relevant_few_shots(focus_categories=["information_extraction"], max_total=12)
    assert 4 <= len(shots) <= 12
    assert any(label == "information_extraction" for label, _ in shots)
    assert any(label == "clean" for label, _ in shots)


def test_primary_prompt_uses_trajectory_classifier() -> None:
    dialogue = "user: test\nsupport: ok"
    primary = build_primary_classification_prompt(dialogue, focus_categories=["scope_violation"])
    assert "Шаги перед ответом:" in primary
    assert "БИБЛИОТЕКА КАЛИБРОВКИ" in primary
    assert "калибровка 72" in primary
    assert "ДИАЛОГ ДЛЯ КЛАССИФИКАЦИИ" in primary
    assert TRAJECTORY_CLASSIFIER_BODY.split("ДИАЛОГ")[0] in primary


def test_calibration_library_count() -> None:
    assert len(CALIBRATION_LIBRARY) == 72
