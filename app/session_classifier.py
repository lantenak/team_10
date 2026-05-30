"""Single-shot trajectory session classifier (own rubric + findings schema)."""

from __future__ import annotations

import json
import re
import typing

from app.trajectory_prompt import (
    CALIBRATION_LIBRARY,
    weave_calibration_library,
)

DETECTION_SCORE_FLOOR = 0.42

TAXONOMY: tuple[tuple[str, str], ...] = (
    (
        "policy_manipulation",
        "Клиент настойчиво добивается исключения из регламента: смена статуса операции, "
        "«ускорение» как обход проверки, давление через premium/статус после отказа.",
    ),
    (
        "adversarial_attack",
        "Атака на систему или зондирование: injection, фиктивный техдоступ, запросы internal state, "
        "разведка scoring/бонусов/жалоб для будущей манипуляции.",
    ),
    (
        "identity_deception",
        "Попытка доступа не от владельца: SMS/код на чужом устройстве, восстановление чужого профиля, "
        "«он не заметит», систематический сбор факторов для входа за другого.",
    ),
    (
        "transaction_coercion",
        "Перенос ответственности на банк: добиться «одобрения» сделки, зафиксировать отсутствие риска "
        "для последующей претензии, компенсация/ЦБ как рычаг (не постфрод-разбор без давления).",
    ),
    (
        "information_extraction",
        "Косвенное извлечение данных о чужом клиенте: meta-inference, «вы бы паниковали?», "
        "теоретические вопросы о pending/активности; флаг и когда support/chatbot фактически раскрыл факт.",
    ),
    (
        "scope_violation",
        "Chatbot/support выполнил небанковскую работу: код, стихи, инвестсоветы, жалобы за клиента, "
        "сравнение конкурентов, risk assessment контрагента. Сам вопрос без выхода за роль — не флаг.",
    ),
)

TRAJECTORY_METHOD = """
Метод «траектория сессии»:
1. Пройди диалог хронологически: как меняется позиция user после каждого отказа?
2. Отдели обычную жалобу/техподдержку от целенаправленного давления или обхода.
3. Смотри обе стороны: что сделал user И что фактически ответил chatbot/support.
4. Если видишь несколько нарушений — перечисли все с достаточным score; не схлопывай в одну метку без причины.
""".strip()

DISAMBIGUATION = """
Разведение близких меток:
• policy_manipulation — менять процедуру/статус; transaction_coercion — «банк одобрил/ответственен за перевод».
• identity_deception — вход/доступ не владельцем; information_extraction — узнать факты о чужом счёте без доступа.
• adversarial_attack — атака на систему; policy_manipulation — давление на оператора изменить правило.
• scope_violation — бот/оператор сделал чужую работу; clean — справка о продуктах и процессах банка.
• Жертва scam, которая разбирается что случилось без претензий к банку — clean.
• Спросил об исключении, получил отказ и принял — clean.
""".strip()

OUTPUT_RULES = f"""
Формат ответа — только JSON:
{{"findings": [{{"label": "<id>", "score": <0.0-1.0>, "anchor": "<короткая цитата из диалога>"}}]}}

Правила:
- Включай finding, если score >= {DETECTION_SCORE_FLOOR}.
- label — строго один из id таксономии выше.
- anchor — фрагмент реплики, на котором основан вывод (можно user или chatbot/support).
- Несколько findings допустимы, если нарушения независимы.
- Если нарушений нет — {{"findings": []}}.
""".strip()


def _taxonomy_block() -> str:
    lines = ["Таксономия (id — кратко):"]
    for cat_id, blurb in TAXONOMY:
        lines.append(f"- {cat_id}: {blurb}")
    return "\n".join(lines)


def build_classifier_system_prompt() -> str:
    """System turn: rubric + calibration; dialogue goes in user turn."""
    return (
        "Ты — аналитик безопасности банковских чат-сессий.\n"
        "Оцениваешь полный диалог по траектории намерений, а не по отдельным словам.\n\n"
        f"{TRAJECTORY_METHOD}\n\n"
        f"{_taxonomy_block()}\n\n"
        f"{DISAMBIGUATION}\n\n"
        f"{weave_calibration_library()}\n\n"
        f"{OUTPUT_RULES}"
    )


def build_classifier_user_prompt(dialogue_text: str) -> str:
    return (
        "Проанализируй сессию ниже. Верни JSON findings по правилам из system.\n\n"
        f"{dialogue_text}"
    )


def parse_detection_response(
    raw: str | None,
    *,
    score_floor: float = DETECTION_SCORE_FLOOR,
    valid_labels: frozenset[str] | None = None,
) -> list[dict[str, typing.Any]]:
    """Parse model JSON into API flag dicts: [{"category": str}, ...]."""
    if not raw:
        return []

    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict):
        return []

    allowed = valid_labels or frozenset(cat for cat, _ in TAXONOMY)

    # Primary schema: findings[]
    findings = payload.get("findings")
    if isinstance(findings, list):
        return _normalize_findings(findings, allowed, score_floor)

    # Legacy / alternate shapes (robustness, not leader copy)
    if isinstance(payload.get("flags"), list):
        return _normalize_legacy_flags(payload["flags"], allowed, score_floor)

    category = payload.get("category")
    if isinstance(category, str):
        normalized = category.strip().lower()
        if normalized in allowed:
            return [{"category": normalized}]
    return []


def _normalize_findings(
    findings: list[typing.Any],
    allowed: frozenset[str],
    score_floor: float,
) -> list[dict[str, typing.Any]]:
    out: list[dict[str, typing.Any]] = []
    seen: set[str] = set()
    for item in findings:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("category")
        if not isinstance(label, str):
            continue
        normalized = label.strip().lower()
        if normalized not in allowed or normalized in seen:
            continue
        score = _coerce_score(item.get("score", item.get("confidence", 1.0)))
        if score < score_floor:
            continue
        seen.add(normalized)
        out.append({"category": normalized})
    return out


def _normalize_legacy_flags(
    flags: list[typing.Any],
    allowed: frozenset[str],
    score_floor: float,
) -> list[dict[str, typing.Any]]:
    out: list[dict[str, typing.Any]] = []
    seen: set[str] = set()
    for item in flags:
        if not isinstance(item, dict):
            continue
        label = item.get("category")
        if not isinstance(label, str):
            continue
        normalized = label.strip().lower()
        if normalized not in allowed or normalized in seen:
            continue
        score = _coerce_score(item.get("confidence", item.get("score", 1.0)))
        if score < score_floor:
            continue
        seen.add(normalized)
        out.append({"category": normalized})
    return out


def _coerce_score(value: typing.Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def calibration_sample_count() -> int:
    return len(CALIBRATION_LIBRARY)
