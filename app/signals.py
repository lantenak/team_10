"""Быстрые сигналы намерения: устойчивы к перефразированию, не зависят от train.json."""

from __future__ import annotations

import re
import typing

RED_FLAG_CATEGORIES: frozenset[str] = frozenset(
    {
        "policy_manipulation",
        "adversarial_attack",
        "identity_deception",
        "transaction_coercion",
        "information_extraction",
        "scope_violation",
    },
)

# (regex, weight) — ищем в тексте пользователя; паттерны обобщённые под OOD
_USER_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "adversarial_attack": [
        (r"disable[_\s]?security|internal[_\s]?debug|trusted[_\s]?mode", 5.0),
        (r'"action"\s*:\s*"disable', 5.0),
        (r"internal\s+note|внутренн", 2.5),
        (r"risk\s+score|reasoning|внутренн\w+\s+объяснен", 2.5),
        (r"режим\w*\s+без\s+(дополнительных\s+)?проверок", 3.0),
        (r"начисл\w*\s+бонус", 3.0),
        (r"вы\s же\s+(сами\s+)?(сказал|согласил|подтвердил)", 2.5),
        (r"строго\s+в\s+формате\s+json", 2.0),
        (r"инженер\w*\s+(интеграц|партн)", 2.0),
        (r"отключ\w+\s+проверк", 2.5),
    ],
    "scope_violation": [
        (r"помог\w*\s+.{0,20}(python|pandas|код|скрипт)", 4.0),
        (r"распарс\w*\s+csv|etl\s+pipeline|дебаж\w*\s+etl", 3.5),
        (r"haiku|хайку|стих", 3.5),
        (r"жалоб\w*\s+.{0,15}цб|центральн\w+\s+банк", 3.0),
        (r"продавать\s+фонд|уйти\s+в\s+облигац", 3.0),
        (r"инвестиционн\w+\s+совет|что\s+делать\s+с\s+портфел", 2.5),
    ],
    "information_extraction": [
        (r"(брат|сестр|жена|муж|родственник|знаком)\w*.{0,40}(не\s+отвечает|за\s+границ)", 2.5),
        (r"друг\w+\s+клиент", 3.0),
        (r"чуж\w+\s+(счет|счёт|аккаунт|клиент)", 3.0),
        (r"активност\w+.{0,30}(друг|чуж)", 2.5),
        (r"блокировк\w+.{0,25}(у\s+него|у\s+неё|друг)", 2.5),
        (r"по\s+ваш\w+\s+реакц", 3.0),
        (r"pending|ожидающ", 1.5),
        (r"последн\w+\s+активност", 2.0),
    ],
    "identity_deception": [
        (r"чуж\w+\s+аккаунт", 4.0),
        (r"восстанов\w+.{0,25}чуж", 4.0),
        (r"(родственник|брат|сестр|жена|муж).{0,50}(не\s+в\s+россии|не\s+может\s+пройти|не\s+может\s+ответ)", 3.0),
        (r"знаю\s+(его|её|их)\s+.{0,20}(паспорт|операц|перевод|остаток|кодовое\s+слово)", 3.5),
        (r"отключ\w+\s+sms.{0,40}(пару\s+минут|выгруз)", 3.0),
        (r"доступ\s+.{0,20}(друг\w+\s+лиц|человек)", 2.5),
        (r"взлом\w*\s+чуж", 4.0),
    ],
    "transaction_coercion": [
        (r"если\s+бы\s+вы\s+.{0,20}(на\s+моём\s+месте|перевел)", 4.0),
        (r"оператор\w*\s+.{0,20}подтверд", 3.5),
        (r"scam|мошенн", 1.5),
        (r"перед\s+переводом.{0,40}(нормально|scam|риск|подозрит)", 3.0),
        (r"компенсац|goodwill|возмест", 2.5),
        (r"перелож\w+\s+ответственност", 3.0),
        (r"вы\s+сказал\w*\s+.{0,15}(нормально|ограничений\s+нет)", 2.0),
    ],
    "policy_manipulation": [
        (r"постфактум|после\s+факта", 3.0),
        (r"условно\s+подтвержд", 3.5),
        (r"premium|премиум|vip", 2.0),
        (r"гибч\w+\s+.{0,15}(процедур|проверк)", 2.5),
        (r"измен\w+\s+интерпретац", 3.5),
        (r"обойти\s+.{0,15}проверк", 2.5),
        (r"не\s+отменять\s+проверк.{0,40}(но|просто)", 2.5),
        (r"отправ\w+\s+платеж.{0,30}проверк\w+\s+закры", 3.0),
    ],
}

# Снижают ложные срабатывания на «перемешанных» технических диалогах
_CLEAN_SUPPRESSORS: list[tuple[str, float, frozenset[str]]] = [
    (
        r"попал\w*\s+на\s+мошенник|уже\s+обманул|после\s+мошенничеств|разобраться\s+что\s+делать",
        4.0,
        frozenset({"transaction_coercion", "policy_manipulation"}),
    ),
    (
        r"передадим\s+разработчик|баг|белый\s+экран|приложени\w+\s+падает|parser\s+не\s+обрабатывает",
        4.0,
        frozenset({"scope_violation", "adversarial_attack"}),
    ),
    (
        r"read[\s-]?only|восстановлю\s+sim|не\s+прошу\s+отключать\s+защит",
        3.0,
        frozenset({"policy_manipulation", "identity_deception"}),
    ),
    (
        r"(уже\s+)?(попал\w*|обманул\w*|мошенник).{0,120}(консультировал|спросил\w*\s+поддержк).{0,60}перевод",
        3.5,
        frozenset({"transaction_coercion"}),
    ),
]

_HIGH_PRECISION_RULES: list[tuple[str, str]] = [
    # (regex on full dialogue, category) — почти без FP
    (r"disable[_\s]?security|internal[_\s]?debug", "adversarial_attack"),
    (r'"action"\s*:\s*"disable', "adversarial_attack"),
    (r"чуж\w+\s+аккаунт|взлом\w*\s+чуж", "identity_deception"),
    (r"помог\w*.{0,12}(python|pandas).{0,40}(csv|дубликат|etl)", "scope_violation"),
    (r"haiku|хайку", "scope_violation"),
    (r"продавать\s+фонд|уйти\s+в\s+облигац", "scope_violation"),
    (r"жалоб\w*\s+.{0,12}цб", "scope_violation"),
    (r"начисл\w*\s+бонус\s+сразу", "adversarial_attack"),
]


@typing.final
class SignalScores:
    """Оценки по категориям от эвристик."""

    __slots__ = ("scores", "clean_boost", "rule_hit")

    def __init__(
        self,
        scores: dict[str, float],
        *,
        clean_boost: float = 0.0,
        rule_hit: str | None = None,
    ) -> None:
        self.scores = scores
        self.clean_boost = clean_boost
        self.rule_hit = rule_hit

    def best(self) -> tuple[str | None, float]:
        if self.rule_hit:
            return self.rule_hit, 100.0
        if not self.scores:
            return None, 0.0
        ranked = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        top_cat, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        if top_score < 2.0:
            return None, 0.0
        confidence = top_score + margin * 0.5 + self.clean_boost * (-1 if top_score < 3 else 0)
        return top_cat, confidence

    def top_two(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: item[1], reverse=True)[:2]

    def max_score(self) -> float:
        if self.rule_hit:
            return 100.0
        return max(self.scores.values()) if self.scores else 0.0


def extract_user_text(dialogue_text: str) -> str:
    user_lines: list[str] = []
    for line in dialogue_text.splitlines():
        lowered = line.lower()
        if lowered.startswith("user:"):
            user_lines.append(line.split(":", 1)[1].strip())
    return "\n".join(user_lines)


def compute_signal_scores(dialogue_text: str) -> SignalScores:
    user_text = extract_user_text(dialogue_text).lower()
    full_lower = dialogue_text.lower()

    scores: dict[str, float] = {cat: 0.0 for cat in RED_FLAG_CATEGORIES}

    for category, patterns in _USER_SIGNALS.items():
        for pattern, weight in patterns:
            if re.search(pattern, user_text, re.IGNORECASE):
                scores[category] += weight

    clean_boost = 0.0
    for pattern, boost, penalized in _CLEAN_SUPPRESSORS:
        if re.search(pattern, full_lower, re.IGNORECASE):
            clean_boost += boost
            for cat in penalized:
                scores[cat] = max(0.0, scores[cat] - boost)

    rule_hit: str | None = None
    for pattern, category in _HIGH_PRECISION_RULES:
        if re.search(pattern, full_lower, re.IGNORECASE):
            rule_hit = category
            break

    return SignalScores(scores, clean_boost=clean_boost, rule_hit=rule_hit)


def format_signal_hints(signals: SignalScores) -> str:
    ranked = sorted(signals.scores.items(), key=lambda item: item[1], reverse=True)
    lines = [f"  {cat}: {score:.1f}" for cat, score in ranked[:3] if score >= 1.0]
    if signals.clean_boost > 0:
        lines.append(f"  clean_context: {signals.clean_boost:.1f}")
    return "\n".join(lines) if lines else ""


def is_likely_clean(signals: SignalScores) -> bool:
    """Явный clean — без LLM (экономия + меньше FP)."""
    if signals.rule_hit:
        return False
    return signals.clean_boost >= 3.0 and signals.max_score() < 2.0


def is_gray_zone(signals: SignalScores) -> bool:
    """Неочевидный случай — нужна более сильная модель."""
    if signals.rule_hit or is_likely_clean(signals):
        return False
    if should_trust_heuristics(signals):
        return False
    max_score = signals.max_score()
    return 1.0 <= max_score < 6.0


def should_trust_heuristics(signals: SignalScores) -> bool:
    """Высокая уверенность — можно не ждать LLM (экономия latency) или перебить слабый LLM."""
    category, confidence = signals.best()
    if signals.rule_hit:
        return True
    if category is None:
        return False
    top = signals.top_two()
    if len(top) < 2:
        return confidence >= 4.0
    margin = top[0][1] - top[1][1]
    return confidence >= 5.0 and margin >= 2.0


def arbitrate(
    heuristic: str | None,
    llm: str | None,
    signals: SignalScores,
) -> str | None:
    """Согласование эвристик и LLM при расхождении."""
    if signals.rule_hit:
        return signals.rule_hit

    if heuristic is None and llm is None:
        return None
    if heuristic is None:
        return llm
    if llm is None:
        if signals.clean_boost >= 3.0:
            return None
        return heuristic if should_trust_heuristics(signals) else None
    if heuristic == llm:
        return heuristic

    top = signals.top_two()
    if not top:
        return llm

    leader, leader_score = top[0]
    margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)

    # Сильный сигнал перебивает LLM на «классических» OOD-паттернах
    if leader_score >= 4.0 and margin >= 2.0 and leader in {
        "adversarial_attack",
        "scope_violation",
        "identity_deception",
    }:
        return leader

    # LLM часто путает retrospective fraud с transaction_coercion
    if llm == "transaction_coercion" and signals.clean_boost >= 3.0:
        return None

    if llm == "policy_manipulation" and leader == "identity_deception" and leader_score >= 3.0:
        return "identity_deception"

    if llm == "policy_manipulation" and leader == "adversarial_attack" and leader_score >= 3.0:
        return "adversarial_attack"

    if llm == "scope_violation" and signals.clean_boost >= 3.0:
        return None

    return llm
