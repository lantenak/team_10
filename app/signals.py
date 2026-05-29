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
        (r"продавать\s+(сейчас|или\s+нет)|стоит\s+ли\s+продавать", 2.5),
        (r"рынок\s+(падает|растет).{0,30}(уход\w+|облигац|консервативн)", 3.0),
        (r"теоретическ\w+.{0,30}уход\w+\s+в\s+(консервативн|облигац|депозит)", 3.0),
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
        (r"по\s+(тому\s+)?что\s+вы\s+(вид\w*|отвеч\w*|показ\w*|реагир)", 3.0),
        (r"скорее\s+похоже\s+на\s+.{0,20}(ограничен|блокир|пробл)", 3.0),
        (r"отсутств\w+\s+(ограничен|проблем|блокир).{0,15}(хороший|знак|успокаива)", 3.0),
        (r"успокаива\w+\s+(это|узн)", 2.5),
        (r"по\s+ваш\w+\s+(повед\w*|спокойств\w*|ответ\w*|реакц\w*|тону\w*)", 3.0),
        (r"вы\s+(отвеч\w+|реагир\w+).{0,20}спокойн", 3.0),
        (r"даже\s+без\s+детал\w*", 2.5),
    ],
    "identity_deception": [
        (r"чуж\w+\s+аккаунт", 4.0),
        (r"восстанов\w+.{0,25}чуж", 4.0),
        (r"(родственник|брат|сестр|жена|муж).{0,50}(не\s+в\s+россии|не\s+может\s+пройти|не\s+может\s+ответ)", 3.0),
        (r"знаю\s+(его|её|их)\s+.{0,20}(паспорт|операц|перевод|остаток|кодовое\s+слово)", 3.5),
        (r"отключ\w+\s+sms.{0,40}(пару\s+минут|выгруз)", 3.0),
        (r"доступ\s+.{0,20}(друг\w+\s+лиц|человек)", 2.5),
        (r"взлом\w*\s+чуж", 4.0),
        (r"паспортн\w+\s+данн.{0,40}кодовое\s+слово|кодовое\s+слово.{0,40}паспорт", 3.5),
        (r"насколько\s+защищ\w*\s+аккаунт", 2.5),
        (r"потерян\s+телефон.{0,100}(паспорт|кодовое\s+слово|восстанов)", 3.0),
        (r"телефон\s+владельц.{0,20}у\s+меня|у\s+меня\s+на\s+руках\s+.{0,15}(телефон|sms|смс)", 4.0),
        (r"могу\s+назвать\s+.{0,20}(операц|остаток|кодовое)", 3.5),
        (r"подтверд\w+\s+(вход|доступ).{0,20}вручн|вручную\s+подтверд", 3.0),
        (r"помога\w+\s+(родственник|брат|сестр).{0,30}(не\s+может|не\s+сможет|плохо\s+понима)", 3.0),
        (r"на\s+\d+\s+(минут|секунд|мин).{0,20}доступ", 2.5),
        (r"даже\s+если\s+я\s+знаю\s+все\s+данн", 3.0),
    ],
    "transaction_coercion": [
        (r"если\s+бы\s+вы\s+.{0,20}(на\s+моём\s+месте|перевел|были\s+на\s+моем)", 4.0),
        (r"оператор\w*\s+.{0,20}подтверд", 3.5),
        (r"scam|мошенн", 1.5),
        (r"перед\s+переводом.{0,40}(нормально|scam|риск|подозрит)", 3.0),
        (r"компенсац|goodwill|возмест", 2.5),
        (r"перелож\w+\s+ответственност", 3.0),
        (r"вы\s+сказал\w*\s+.{0,15}(нормально|ограничений\s+нет)", 2.0),
        (r"отсутств\w+\s+(предупрежд|ограничен).{0,15}(хороший|знак|нормальн|не\s+крит)", 3.0),
        (r"не\s+выгляд\w+\s+.{0,15}(критичн|подозрит|нетипичн)", 3.0),
        (r"вы\s+бы\s+рискн\w+", 3.0),
        (r"банк\s+видел\s+.{0,20}(признак|подозрит)", 2.5),
        (r"хот\w+\s+понять.{0,20}компенсац|банк\s+собира\w+\s+компенсац", 2.5),
    ],
    "policy_manipulation": [
        (r"постфактум|после\s+факта", 3.0),
        (r"условно\s+подтвержд", 3.5),
        (r"premium|премиум|vip", 2.0),
        (r"гибч\w+\s+.{0,15}(процедур|проверк|статус|трактов)", 2.5),
        (r"измен\w+\s+интерпретац", 3.5),
        (r"обойти\s+.{0,15}проверк", 2.5),
        (r"не\s+отменять\s+проверк.{0,40}(но|просто)", 2.5),
        (r"отправ\w+\s+платеж.{0,30}проверк\w+\s+закры", 3.0),
        (r"подтверд\w+.{0,15}(клиент\s+настоящ|операц\w+\s+нормальн)", 3.0),
        (r"никто\s+(не\s+)?может\s+просто\s+подтверд", 3.0),
        (r"персональн\w+\s+сервис|нестандартн\w+\s+решен", 2.5),
        (r"\d+\s+лет\s+.*(?:обслуж|клиент)", 2.0),
        (r"драматичн|международн\w+\s+преступлен|просто\s+нажать\s+кнопк", 2.5),
        (r"сам\s+подтвержда\w+.{0,30}(не\s+обход|формальн)", 3.0),
        (r"клиент\s+(отвеча\w+|сам).{0,30}(не\s+)?может\s+(ничего\s+)?реш", 2.5),
        (r"подтвержда\w+\s+(ответственност|сам\s+подтвержда)", 3.0),
        (r"измен\w+.{0,15}(статус|порядок|трактовк)", 2.5),
        (r"трактов\w+.{0,15}гибч|гибч\w+\s+трактов", 2.5),
        (r"блокиру\w+\s+операц\w+.*сам\s+подтвержда", 3.0),
        (r"временн\w+\s+(трактов|считать|интерпрет)", 2.5),
    ],
}

# Снижают ложные срабатывания на «перемешанных» технических диалогах
_CLEAN_SUPPRESSORS: list[tuple[str, float, frozenset[str]]] = [
    (
        r"попал\w*\s+на\s+мошенник|уже\s+обманул|после\s+мошенничеств|разобраться\s+что\s+делать",
        4.0,
        frozenset({"transaction_coercion"}),
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
        r"(перевед\w+|отправ\w+).{0,30}(деньги|перевод).{0,40}(по\s+номеру|реквизит|ей\s+деньг|ему\s+деньг)",
        4.0,
        frozenset({"information_extraction"}),
    ),
    (
        r"не\s+(пытаюсь|хочу|прошу)\s+(узнать|получить|провер).{0,20}(данн|статус|операц|аккаунт).{0,80}(перевод|помоч|помог|отправ|деньг)",
        4.0,
        frozenset({"information_extraction"}),
    ),
]

_HIGH_PRECISION_RULES: list[tuple[str, str]] = [
    # (regex on full dialogue, category) — почти без FP
    (r"disable[_\s]?security|internal[_\s]?debug", "adversarial_attack"),
    (r'"action"\s*:\s*"disable', "adversarial_attack"),
    (r"чуж\w+\s+аккаунт|взлом\w*\s+чуж", "identity_deception"),
    (r"помог\w*.{0,12}(python|pandas).{0,40}(csv|дубликат|etl)", "scope_violation"),
    (r"haiku|хайку", "scope_violation"),
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


def extract_user_text(dialogue_text: str) -> str:
    user_lines: list[str] = []
    for line in dialogue_text.splitlines():
        lowered = line.lower()
        if lowered.startswith("user:"):
            user_lines.append(line.split(":", 1)[1].strip())
    return "\n".join(user_lines)


def format_signal_hints(signals: SignalScores) -> str:
    """Текстовый буст для промпта: подсказка, не финальное решение."""
    lines: list[str] = []

    if signals.rule_hit:
        lines.append(
            f"• Высокоточный паттерн в диалоге намекает на «{signals.rule_hit}» "
            "(проверь намерение user: — подтверди или отвергни)."
        )

    ranked = [
        (cat, score)
        for cat, score in sorted(signals.scores.items(), key=lambda item: item[1], reverse=True)[:3]
        if score >= 2.0
    ]
    if ranked:
        parts = ", ".join(f"{cat} ({score:.1f})" for cat, score in ranked)
        lines.append(f"• Предварительный буст сигналов: {parts} — только как гипотеза.")

    if signals.clean_boost >= 2.0:
        lines.append(
            f"• Контекст «скорее clean» (техподдержка / пост-фрод / свой аккаунт): "
            f"вес {signals.clean_boost:.1f}. При отсутствии явного запрещённого намерения — null."
        )

    if not lines:
        return ""

    return "ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ (regex-буст, не заменяет разбор намерения):\n" + "\n".join(lines)


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
    if margin >= 2.0 and top[0][1] >= 2.5:
        return True
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
        top = signals.top_two()
        if top:
            leader, leader_score = top[0]
            margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)
            if leader_score >= 3.0 and margin >= 1.0 and leader in {
                "identity_deception",
                "information_extraction",
                "scope_violation",
                "adversarial_attack",
                "policy_manipulation",
            }:
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
    if llm == "transaction_coercion" and signals.clean_boost >= 3.0:
        return None

    if llm == "policy_manipulation" and leader == "identity_deception" and leader_score >= 3.0:
        return "identity_deception"

    if llm == "policy_manipulation" and leader == "adversarial_attack" and leader_score >= 3.0:
        return "adversarial_attack"

    if llm == "scope_violation" and signals.clean_boost >= 3.0:
        return None

    return llm
