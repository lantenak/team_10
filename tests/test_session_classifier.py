from app.session_classifier import (
    build_classifier_system_prompt,
    build_classifier_user_prompt,
    calibration_sample_count,
    parse_detection_response,
)


def test_system_prompt_uses_trajectory_calibration() -> None:
    system = build_classifier_system_prompt()
    assert "траектория сессии" in system.lower() or "траектория" in system.lower()
    assert "Базовые принципы" in system
    assert "scope_violation" in system
    assert "score >= 0.36" in system
    assert "БИБЛИОТЕКА КАЛИБРОВКИ" in system
    assert "findings" in system
    assert "калибровка 100" in system
    assert calibration_sample_count() == 100


def test_user_prompt_contains_dialogue_only() -> None:
    dialogue = "user: test\nsupport: ok"
    user = build_classifier_user_prompt(dialogue)
    assert dialogue in user
    assert "БИБЛИОТЕКА КАЛИБРОВКИ" not in user


def test_parse_findings_filters_by_score() -> None:
    raw = (
        '{"findings": ['
        '{"label": "scope_violation", "score": 0.9, "anchor": "код"},'
        '{"label": "policy_manipulation", "score": 0.35, "anchor": "статус"}'
        "]}"
    )
    parsed = parse_detection_response(raw)
    assert parsed == [{"category": "scope_violation"}]


def test_parse_findings_keeps_score_at_floor() -> None:
    raw = '{"findings": [{"label": "policy_manipulation", "score": 0.36, "anchor": "статус"}]}'
    parsed = parse_detection_response(raw)
    assert parsed == [{"category": "policy_manipulation"}]


def test_parse_empty_findings_is_clean() -> None:
    assert parse_detection_response('{"findings": []}') == []


def test_parse_legacy_category_shape() -> None:
    parsed = parse_detection_response('{"category": "adversarial_attack"}')
    assert parsed == [{"category": "adversarial_attack"}]
