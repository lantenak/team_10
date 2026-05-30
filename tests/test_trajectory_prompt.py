from app.trajectory_prompt import (
    CALIBRATION_LIBRARY,
    TrajectorySample,
    assemble_trajectory_classifier_prompt,
    weave_calibration_library,
)


def test_calibration_library_size() -> None:
    assert len(CALIBRATION_LIBRARY) == 72
    assert isinstance(CALIBRATION_LIBRARY[0], TrajectorySample)


def test_weave_uses_trajectory_layout_not_category_blocks() -> None:
    block = weave_calibration_library()
    assert "БИБЛИОТЕКА КАЛИБРОВКИ" in block
    assert "калибровка 01" in block
    assert "ожидаемый JSON:" in block
    assert "## Класс:" not in block
    assert "ФЛАГ ДА" not in block
    assert "→ FLAGS" not in block


def test_full_prompt_has_preamble_and_footer() -> None:
    prompt = assemble_trajectory_classifier_prompt()
    assert "Шаги перед ответом:" in prompt
    assert "Допустимые метки" in prompt
    assert "доминирующую метку" in prompt
