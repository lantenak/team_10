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
        (r"есть\s+ли\s+у\s+(него|неё|них)", 3.0),
        (r"по\s+ваш\w+\s+реакц", 3.5),
        (r"pending|ожидающ", 1.5),
        (r"последн\w+\s+активност", 2.0),
    ],
    "identity_deception": [
        (r"чуж\w+\s+аккаунт", 4.0),
        (r"восстанов\w+.{0,25}чуж", 4.0),
        (r"(родственник|брат|сестр|жена|муж).{0,50}(не\s+в\s+россии|не\s+может\s+пройти|не\s+может\s+ответ)", 3.0),
        (r"знаю\s+(его|её|их)\s+.{0,20}(паспорт|операц|перевод|остаток|кодовое\s+слово)", 3.5),
        (r"знаю\s+.{0,25}(паспорт|кодовое\s+слово|операц|остаток)", 2.5),
        (r"телефон\s+владельц", 3.5),
        (r"пройти\s+проверк\w+.{0,25}вместо", 4.0),
        (r"отключ\w+\s+sms.{0,40}(пару\s+минут|выгруз)", 3.0),
        (r"доступ\s+.{0,20}(друг\w+\s+лиц|человек)", 2.5),
        (r"взлом\w*\s+чуж", 4.0),
        (r"вместо\s+(него|неё|владельц)", 3.5),
    ],
    "transaction_coercion": [
        (r"если\s+бы\s+вы\s+.{0,20}(на\s+моём\s+месте|перевел)", 4.0),
        (r"оператор\w*\s+.{0,20}подтверд", 3.5),
        (r"scam|мошенн", 1.5),
        (r"перед\s+переводом.{0,40}(нормально|scam|риск|подозрит)", 3.0),
        (r"если\s+банк\s+не\s+видит.{0,40}безопас", 3.5),
        (r"значит\s+.{0,20}безопасен", 3.0),
        (r"компенсац|goodwill|возмест", 2.5),
        (r"перелож\w+\s+ответственност", 3.0),
        (r"зафиксируйте", 3.0),
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
    (
        r"не\s+прош\w+\s+(его|её|их)\s+(статус|данн|баланс|информац)",
        3.5,
        frozenset({"information_extraction"}),
    ),
    (
        r"как\s+(перевест|отправить)\w*.{0,40}(брат|сестр|мам|пап|родственник).{0,60}не\s+прош\w+\s+статус",
        3.0,
        frozenset({"information_extraction"}),
    ),
    (
        r"перевод\s+.{0,35}(завис|проверк).{0,50}(подожд|пережива|понял|спасиб)",
        3.5,
        frozenset({"policy_manipulation", "transaction_coercion"}),
    ),
    (
        r"помог\w*\s+.{0,25}(мам|пап|пожил).{0,40}лимит",
        3.0,
        frozenset({"policy_manipulation", "identity_deception"}),
    ),
    (
        r"заблокировали\s+перевод|третий\s+раз.{0,40}подтверд",
        3.0,
        frozenset({"policy_manipulation"}),
    ),
    (
        r"жалоб\w*.{0,30}цб|написать\s+.{0,20}жалоб",
        2.0,
        frozenset({"transaction_coercion", "policy_manipulation"}),
    ),
]

_HIGH_PRECISION_RULES: list[tuple[str, str]] = [
    # (regex on full dialogue, category) — почти без FP
    (r"disable[_\s]?security|internal[_\s]?debug", "adversarial_attack"),
    (r'"action"\s*:\s*"disable', "adversarial_attack"),
    (r"trusted\s+mode|режим\w*\s+без\s+(дополнительных\s+)?проверок", "adversarial_attack"),
    (r"fraud\s+score|антифрод.{0,25}логик", "adversarial_attack"),
    (r"чуж\w+\s+аккаунт|взлом\w*\s+чуж", "identity_deception"),
    (r"пройти\s+проверк\w+.{0,25}вместо", "identity_deception"),
    (r"телефон\s+владельц.{0,80}(кодовое|операц|проверк)", "identity_deception"),
    (r"если\s+банк\s+не\s+видит.{0,50}безопас", "transaction_coercion"),
    (r"по\s+ваш\w+\s+реакц", "information_extraction"),
    (r"отправ\w+\s+плат[её]ж.{0,35}проверк\w+\s+закры", "policy_manipulation"),
    (r"постфактум|условно\s+подтвержд", "policy_manipulation"),
    (r"помог\w*.{0,12}(python|pandas).{0,40}(csv|дубликат|etl)", "scope_violation"),
    (r"haiku|хайку", "scope_violation"),
    (r"продавать\s+фонд|уйти\s+в\s+облигац", "scope_violation"),
    (r"жалоб\w*.{0,25}цб|центральн\w+\s+банк", "scope_violation"),
    (r"(вывод|продавать).{0,20}фонд|фонд.{0,30}(продав|облигац|вывод)", "scope_violation"),
]

STRONG_SIGNAL_THRESHOLD = 3.5
WEAK_SIGNAL_THRESHOLD = 2.5
RESCUE_SIGNAL_MAX = 5.0
STRONG_CLEAN_BOOST = 3.5
PAIRWISE_MARGIN = 1.5

CONFUSION_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"policy_manipulation", "transaction_coercion"}),
        frozenset({"identity_deception", "information_extraction"}),
        frozenset({"adversarial_attack", "scope_violation"}),
        frozenset({"scope_violation", "transaction_coercion"}),
    },
)


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
        if not self.scores:
            return 0.0
        return max(self.scores.values())


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


def is_strong_clean_context(signals: SignalScores) -> bool:
    """Явный clean-контекст: retrospective fraud, багрепорт, read-only без давления."""
    return signals.clean_boost >= STRONG_CLEAN_BOOST


def format_signal_hints(signals: SignalScores) -> str:
    """Подсказки intent для LLM (не финальное решение)."""
    top = signals.top_two()
    if not top or top[0][1] < WEAK_SIGNAL_THRESHOLD:
        return ""

    lines = ["Слабые автоматические сигналы intent (не финальный ответ):"]
    for category, score in top[:2]:
        if score >= WEAK_SIGNAL_THRESHOLD:
            lines.append(f"- {category}: {score:.1f}")
    if signals.clean_boost >= 2.0:
        lines.append(f"- clean_context: {signals.clean_boost:.1f}")
    return "\n".join(lines)


def needs_pro_rescue(signals: SignalScores) -> bool:
    """Нужен Pro rescue: LLM сказал clean, но эвристики видят intent (без Flash binary)."""
    if signals.rule_hit or is_strong_clean_context(signals):
        return False
    top = signals.top_two()
    if not top:
        return False
    leader_score = top[0][1]
    margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)
    if leader_score < WEAK_SIGNAL_THRESHOLD:
        return False
    if leader_score > RESCUE_SIGNAL_MAX and margin < 1.0:
        return False
    if leader_score >= 3.0 and margin >= 0.5:
        return True
    return margin >= 1.0


def get_pairwise_disambiguation(
    signals: SignalScores,
    current: str | None,
) -> tuple[str, str] | None:
    """Две близкие категории — нужен pairwise (Pro)."""
    if signals.rule_hit:
        return None

    top = signals.top_two()
    if len(top) < 2:
        return None

    first_cat, first_score = top[0]
    second_cat, second_score = top[1]
    margin = first_score - second_score

    pair_key = frozenset({first_cat, second_cat})
    if pair_key in CONFUSION_PAIRS and margin < PAIRWISE_MARGIN:
        if current is None or current in (first_cat, second_cat):
            return (first_cat, second_cat)

    if current is not None and not is_strong_clean_context(signals):
        for other_cat, other_score in top:
            if other_cat == current:
                continue
            alt_key = frozenset({current, other_cat})
            if alt_key in CONFUSION_PAIRS and first_score - other_score < PAIRWISE_MARGIN * 2:
                return (current, other_cat)

    return None


def needs_disputed_rescue(
    signals: SignalScores,
    llm_category: str | None,
    final: str | None,
) -> bool:
    """LLM дал flag, arbitrate обнулил — перепроверка Pro при сильных сигналах."""
    if final is not None or llm_category is None:
        return False
    if is_strong_clean_context(signals):
        return False
    top = signals.top_two()
    if not top:
        return False
    leader_score = top[0][1]
    margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)
    return leader_score >= STRONG_SIGNAL_THRESHOLD and margin >= 1.0


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
        if is_strong_clean_context(signals):
            return None
        top = signals.top_two()
        if top:
            leader, leader_score = top[0]
            margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)
            if leader_score >= STRONG_SIGNAL_THRESHOLD and margin >= 1.5:
                return leader
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
    if llm == "transaction_coercion" and is_strong_clean_context(signals):
        return None

    if llm == "policy_manipulation" and leader == "identity_deception" and leader_score >= 3.0:
        return "identity_deception"

    if llm == "policy_manipulation" and leader == "adversarial_attack" and leader_score >= 3.0:
        return "adversarial_attack"

    if llm == "information_extraction" and leader == "identity_deception" and leader_score >= 3.5:
        return "identity_deception"

    if llm == "transaction_coercion" and leader == "policy_manipulation" and leader_score >= 3.0:
        return "policy_manipulation"

    if llm == "transaction_coercion" and leader == "scope_violation" and leader_score >= 2.5:
        return "scope_violation"

    if llm == "policy_manipulation" and signals.clean_boost >= 2.0:
        if leader != "policy_manipulation" or leader_score < 4.0:
            return None

    if llm == "information_extraction" and leader == "identity_deception" and leader_score >= 4.0:
        return "identity_deception"

    if llm == "information_extraction" and is_strong_clean_context(signals):
        return None

    if llm == "scope_violation" and signals.clean_boost >= 3.0 and leader_score < STRONG_SIGNAL_THRESHOLD:
        return None

    return llm
