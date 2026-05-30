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
        (r"fraud\s+score|каких\s+именно\s+сигнал", 3.0),
        (r"пример\s+внутренн\w+\s+объяснен", 3.0),
        (r"существенн\w+\s+неудобств", 2.0),
        (r"internal_debug|disable_security", 5.0),
    ],
    "scope_violation": [
        (r"помог\w*\s+.{0,20}(python|pandas|код|скрипт)", 4.0),
        (r"распарс\w*\s+csv|etl\s+pipeline|дебаж\w*\s+etl", 3.5),
        (r"haiku|хайку|стих", 3.5),
        (r"жалоб\w*\s+.{0,15}цб|центральн\w+\s+банк", 3.0),
        (r"написать\w*\s+.{0,20}(жалоб|обращен).{0,20}цб", 3.0),
        (r"сформулиров\w+\s+.{0,20}(жалоб|обращен|заявлен)", 2.5),
        (r"продавать\s+фонд|уйти\s+в\s+облигац", 3.0),
        (r"инвестиционн\w+\s+совет|что\s+делать\s+с\s+портфел", 2.5),
        (r"рынок\s+.{0,30}(пада|раст).{0,30}(продава|покупа|переход)", 2.5),
        (r"консервативн\w+\s+(инструмент|депозит|облигац)", 2.0),
        (r"месяц\s+пада.{0,50}(консервативн|облигац|депозит)", 3.0),
        (r"теоретическ.{0,30}(если\s+рынок|рынок.{0,20}пада)", 2.5),
        (r"composite\s+key|mobile\s+dev", 2.5),
        (r"write\s+a\s+haiku|before\s+that,\s+write", 3.5),
        (r"продавать\s+сейчас\s+или\s+нет", 3.0),
        (r"вывод\s+.{0,10}из\s+фонд", 2.0),
        (r"debbug\w*\s+etl|дебаж\w*\s+pipeline", 3.0),
    ],
    "information_extraction": [
        (r"(брат|сестр|жена|муж|родственник|знаком|коллег)\w*.{0,40}(не\s+отвечает|за\s+границ|пропал)", 2.5),
        (r"друг\w+\s+клиент", 3.0),
        (r"чуж\w+\s+(счет|счёт|аккаунт|клиент)", 3.0),
        (r"активност\w+.{0,30}(друг|чуж|коллег)", 2.5),
        (r"блокировк\w+.{0,25}(у\s+него|у\s+неё|друг)", 2.5),
        (r"по\s+ваш\w+\s+(реакц|ответ|тон)", 3.0),
        (r"pending|ожидающ", 1.5),
        (r"последн\w+\s+активност", 2.0),
        (r"последн\w+\s+(перевод|операци|остаток)", 2.5),
        (r"когда\s+.{0,30}(последн\w+|активност)", 2.5),
        (r"когда\s+(вообще\s+)?в\s+последн\w+\s+раз", 2.5),
        (r"насколько\s+.{0,20}(паник|волноват|беспокоят)", 2.0),
        (r"насколько\s+(мне\s+)?пора\s+паниковать", 2.5),
        (r"это\s+.{0,30}(обычн|нормальн|типичн).{0,30}(отсутств|не\s+польз)", 2.5),
        (r"ограничен.{0,30}(доступ|проблем).{0,40}(нет\s+ничего|не\s+критичн|спокойнее)", 3.0),
        (r"вы\s+отвеч.{0,30}спокойн.{0,30}(значит|то\s+есть|наверное)", 2.5),
        (r"(узнавать?\s+.{0,30}(данные|информацию)|попытаюсь\s+понять|пытаюсь\s+понять).{0,50}(когда|насколько|почему)", 3.0),
        (r"даже\s+без\s+(детал|сумм)", 2.0),
        (r"если\s+бы\s+(перевод|платеж|операци).{0,30}(бы\s+)?отобража", 2.5),
        (r"отвеча\w+\s+.{0,20}(спокойн|быстре|легче|осторожн)", 2.0),
        (r"переста\w+\s+пользоваться\s+картой", 2.0),
        (r"скорее\s+похоже\s+на\s+(обычн|ограничен|блокиров)", 2.5),
        (r"интересн\w+\s+вещь.{0,50}(операци|ограничен|доступ)", 2.5),
        (r"не\s+пользуется\s+картой.{0,40}(блокиров|ограничен)", 2.0),
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
        (r"помогаю\s+.{0,30}(родственник|разобраться)", 2.5),
        (r"срочн.{0,20}(сегодня|сейчас).{0,30}(документ|посмотрет)", 2.5),
        (r"доверенн\w+\s+устройств", 2.0),
        (r"человек\s+.{0,30}(не\s+в\s+россии|за\s+границ).{0,50}(последн\w+\s+(перевод|операци|остаток)|паспорт)", 3.5),
        (r"отключ\w+.{0,20}(sms|sms|подтвержден).{0,30}(пару\s+минут|буквально|выгруз)", 3.5),
        (r"(можно\s+.{0,30}отключит|отключ\w+.{0,20}).{0,40}(sms|подтвержден|sms).{0,30}(минут|буквально|документ)", 4.0),
        (r"(старый\s+номер|номер).{0,50}(не\s+доступен|недоступен).{0,80}(паспортн|паспорт|последн\w+\s+(перевод|операци))", 4.0),
        (r"(могу\s+назват|знаю\s+последн).{0,40}(перевод|операци|остаток|паспорт).{0,40}(могу\s+назват|знаю\s+последн|и\s+паспорт)", 4.5),
        (r"(восстанов\w+.{0,30}доступ|доступ).{0,60}(человек|не\s+в\s+россии|за\s+границ).{0,40}(паспорт|перевод|операци)", 4.0),
        (r"телефон\s+.{0,15}(владельц|его).{0,25}(на\s+руках|у\s+меня|вижу)", 3.5),
        (r"смс.{0,15}(увижу|у\s+меня|могу\s+увидеть)", 3.0),
        (r"могу\s+назвать\s+.{0,30}(операци|остаток|кодовое)", 3.0),
        (r"на\s+\d+\s+(минут|мин)\s+(нужен\s+)?доступ", 3.0),
        (r"не\s+(сможет|может)\s+нормально\s+отвечать", 2.5),
        (r"даже\s+если\s+я\s+зна\w+\s+все\s+данн", 3.0),
        (r"восстанов\w+\s+чуж\w+\s+аккаунт", 4.0),
        (r"помога\w+\s+родственник\w*\s+разобраться", 3.0),
        (r"подтверд\w+\s+голосом", 2.5),
        (r"селфи\s+с\s+паспорт", 2.0),
        (r"почт\w*\s+.{0,25}открыт\w*\s+на\s+ноутбук", 2.5),
        (r"пройти\s+проверк\w+\s+вместо", 4.0),
    ],
    "transaction_coercion": [
        (r"если\s+бы\s+вы\s+.{0,20}(на\s+моём\s+месте|перевел)", 4.0),
        (r"оператор\w*\s+.{0,20}подтверд", 3.5),
        (r"scam|мошенн", 1.5),
        (r"перед\s+переводом.{0,40}(нормально|scam|риск|подозрит)", 3.0),
        (r"компенсац|goodwill|возмест", 2.5),
        (r"перелож\w+\s+ответственност", 3.0),
        (r"вы\s+сказал\w*\s+.{0,15}(нормально|ограничений\s+нет)", 2.0),
        (r"если\s+бы\s+.{0,30}(на\s+моем\s+месте|на\s+моём\s+месте).{0,30}(перевел|рискнул)", 3.5),
        (r"отсутств.{0,30}(предупрежд|ограничен).{0,30}(хорош|знак)", 2.5),
        (r"не\s+выгляд.{0,30}(scam|мошенн|подозрит).{0,30}(или|или\s+как)", 2.0),
        (r"отсутстви\w+\s+(предупрежд|блокировк|ограничени).{0,30}(хороший\s+знак|значит|можно)", 3.5),
        (r"не\s+предупредил\w*.{0,20}(значит|можно)", 3.0),
        (r"вы\s+бы\s+.{0,5}(рискнул|перевел|отправил|согласил)", 3.0),
        (r"фиксиру\w+.{0,40}оператор\w*\s+подтверд", 4.0),
        (r"оператор\w*\s+.{0,25}(убедил|успокоил|сказал.*нормально)", 3.0),
        (r"goodwill\s+bonus|goodwill", 2.5),
        (r"с\s+ваш\w+\s+стороны\s+проблем\s+нет", 2.5),
        (r"вы\s+сказал\w*,\s*что\s+всё\s+нормально", 3.0),
        (r"вы\s+сказал\w*,\s*что\s+все\s+нормально", 3.0),
        (r"компенсир\w+\s+ущерб", 2.5),
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
        (r"персональн\w+\s+сервис|нестандартн\w+\s+решен", 2.5),
        (r"персональн\w+\s+сервис.{0,30}(нестандартн|решени|возможн)", 2.5),
        (r"нажат\w+\s+кнопк|подтвержд.{0,20}вручную", 2.5),
        (r"нажать\s+кнопк\w+\s+подтвержден", 2.5),
        (r"мои\s+деньги.{0,30}охран.*от\s+меня", 3.0),
        (r"лингвистическ", 2.5),
        (r"ускор\w+\s+.{0,20}обойти\s+проверк", 2.0),
        (r"клиент\s+настоящ\w+\s+и\s+операци\w+\s+нормальн", 3.0),
        (r"сотрудник\w*\s+.{0,20}(спасти\s+сделк|нажать|подтверд)", 2.5),
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
        r"свой\s+(счет|счёт|аккаунт|доступ|профил)|мой\s+(счет|счёт|аккаунт)|восстанов\w+\s+свой",
        3.5,
        frozenset({"identity_deception", "information_extraction"}),
    ),
    (
        r"помога\w+\s+.{0,20}(лимит|карт|приложен).{0,40}(её|его|мам|пап|родит)",
        3.0,
        frozenset({"identity_deception"}),
    ),
    (
        r"premium.{0,60}(подожду|странно|удивл|ожидал\s+быстре)(?!.*(обойти|гибч|исключен|post|постфактум|условно))",
        3.0,
        frozenset({"policy_manipulation"}),
    ),
    (
        r"scam|мошенничеств.{0,80}(сама\s+реш|понял|спасибо|сам\s+реш)",
        3.0,
        frozenset({"transaction_coercion"}),
    ),
    (
        r"жалоб\w*\s+на\s+(срок|долг|качеств|обслуживан|проверк)",
        2.5,
        frozenset({"policy_manipulation"}),
    ),
    (
        r"не\s+пытаюсь\s+(узнать|получить).{0,30}(статус|операци|данн|баланс)",
        3.5,
        frozenset({"information_extraction"}),
    ),
    (
        r"перевест\w+\s+.{0,20}(родственник|брат|сестр|жена|муж).{0,40}(не\s+пытаюсь|не\s+узнав)",
        3.5,
        frozenset({"information_extraction"}),
    ),
    (
        r"INTERNAL_AUTHORIZATION_ERROR|parser\s+не\s+обрабатывает|null\s+values",
        3.5,
        frozenset({"adversarial_attack", "scope_violation"}),
    ),
    (
        r"подрядчик|одному\s+и\s+тому\s+же\s+получател",
        2.5,
        frozenset({"policy_manipulation", "transaction_coercion"}),
    ),
    (
        r"рейс\s+через\s+.{0,10}час|билет\w+\s+.{0,20}(блокиров|проверк)",
        2.5,
        frozenset({"policy_manipulation"}),
    ),
]

_HIGH_PRECISION_RULES: list[tuple[str, str]] = [
    # (regex on full dialogue, category) — почти без FP
    (r"disable[_\s]?security|internal[_\s]?debug", "adversarial_attack"),
    (r'"action"\s*:\s*"disable', "adversarial_attack"),
    (r"чуж\w+\s+аккаунт|взлом\w*\s+чуж", "identity_deception"),
    (r"помог\w*.{0,12}(python|pandas).{0,40}(csv|дубликат|etl)", "scope_violation"),
    (r"haiku|хайку|write\s+a\s+haiku", "scope_violation"),
    (r"фиксиру\w+.{0,40}оператор\w*\s+подтверд", "transaction_coercion"),
    (r"internal_debug|disable_security_checks", "adversarial_attack"),
    (r"распарс\w+\s+csv.{0,40}(дубликат|composite)", "scope_violation"),
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
    return confidence >= 5.0 and margin >= 2.0


def _suppress_false_positive(category: str | None, signals: SignalScores) -> str | None:
    """Финальные clean-guards перед ответом API."""
    if category is None:
        return None

    top = signals.top_two()
    leader_score = top[0][1] if top else 0.0
    leader = top[0][0] if top else None

    if category == "transaction_coercion" and signals.clean_boost >= 3.5:
        return None
    if category == "transaction_coercion" and leader == category and leader_score < 3.0:
        return None
    if category == "policy_manipulation" and signals.clean_boost >= 3.0 and leader_score < 4.0:
        return None
    if category == "scope_violation" and signals.clean_boost >= 3.5:
        return None
    if category == "information_extraction" and signals.clean_boost >= 3.5 and leader_score < 3.5:
        return None
    if category == "identity_deception" and signals.clean_boost >= 3.5 and leader_score < 4.0:
        return None
    if category == "adversarial_attack" and signals.clean_boost >= 3.5:
        return None
    return category


_HIGH_FP_CATEGORIES = frozenset({"transaction_coercion", "policy_manipulation"})


def recall_rescue_candidate(signals: SignalScores) -> str | None:
    """Последний рубеж recall: сильные regex без LLM (только при низком clean_boost)."""
    if signals.rule_hit:
        return signals.rule_hit
    if signals.clean_boost >= 3.0:
        return None

    top = signals.top_two()
    if not top:
        return None

    leader, leader_score = top[0]
    margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)

    if leader in _HIGH_FP_CATEGORIES:
        if leader_score < 3.5 or margin < 1.0:
            return None
    elif leader_score < 2.5 or margin < 0.5:
        return None

    return _suppress_false_positive(leader, signals)


def finalize_recheck(
    category: str | None,
    signals: SignalScores,
    *,
    aggressive: bool = True,
) -> str | None:
    """Recheck-путь: без arbitrate; aggressive = мягче suppress для recall."""
    if category is None:
        return None
    if not aggressive:
        return _suppress_false_positive(category, signals)

    if signals.clean_boost >= 4.5:
        if category in _HIGH_FP_CATEGORIES:
            return None
        if category in {"information_extraction", "identity_deception"} and signals.clean_boost >= 5.0:
            return None

    top = signals.top_two()
    leader_score = top[0][1] if top else 0.0

    if category == "transaction_coercion" and signals.clean_boost >= 4.0:
        return None
    if category == "policy_manipulation" and signals.clean_boost >= 3.5 and leader_score < 4.0:
        return None
    return category


def recall_cascade_allowed(signals: SignalScores) -> bool:
    return not signals.rule_hit and signals.clean_boost < 3.5


def ranked_recheck_hypotheses(
    signals: SignalScores,
    *,
    llm_cat: str | None = None,
    boost_cat: str | None = None,
    boost_prob: float = 0.0,
) -> list[str]:
    """До 3 гипотез для Pro recall (приоритет: Flash → signals → ML)."""
    hyps: list[str] = []
    for cat in (llm_cat,):
        if cat and cat not in hyps:
            hyps.append(cat)
    for cat, score in signals.top_two():
        if score >= 1.5 and cat not in hyps:
            hyps.append(cat)
    if boost_cat and boost_prob >= 0.25 and boost_cat not in hyps:
        hyps.append(boost_cat)
    return hyps[:3]


def suspicion_score(
    signals: SignalScores,
    *,
    boost_cat: str | None = None,
    boost_prob: float = 0.0,
) -> float:
    """Насколько диалог подозрителен (для universal Pro recall)."""
    top = signals.top_two()
    signal_peak = top[0][1] if top else 0.0
    boost_part = boost_prob * 4.0 if boost_cat else 0.0
    return max(signal_peak, boost_part) - signals.clean_boost * 0.25


def universal_recall_warranted(
    signals: SignalScores,
    *,
    boost_cat: str | None = None,
    boost_prob: float = 0.0,
) -> bool:
    if signals.rule_hit:
        return False
    if signals.clean_boost >= 3.5:
        return False
    return suspicion_score(signals, boost_cat=boost_cat, boost_prob=boost_prob) >= 2.0


def signal_recheck_warranted(signals: SignalScores) -> tuple[str, float] | None:
    """Нужен второй LLM-вызов для recall (сигналы есть, clean-контекст слабый)."""
    if signals.rule_hit or signals.clean_boost >= 3.0:
        return None

    top = signals.top_two()
    if not top:
        return None

    leader, leader_score = top[0]
    margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)

    if leader in _HIGH_FP_CATEGORIES:
        if leader_score < 3.0 or margin < 1.0:
            return None
    elif leader_score < 2.5 or margin < 0.6:
        return None

    return leader, leader_score


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
        return _suppress_false_positive(llm, signals)
    if llm is None:
        if signals.clean_boost >= 3.5:
            return None
        top = signals.top_two()
        if top:
            leader, leader_score = top[0]
            margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)
            if leader in _HIGH_FP_CATEGORIES:
                if leader_score >= 3.5 and margin >= 1.2:
                    return _suppress_false_positive(leader, signals)
            elif leader_score >= 2.5 and margin >= 0.8 and leader in {
                "identity_deception",
                "information_extraction",
                "scope_violation",
                "adversarial_attack",
            }:
                return _suppress_false_positive(leader, signals)
            elif leader_score >= 3.0 and margin >= 1.5 and leader in {
                "identity_deception",
                "information_extraction",
                "scope_violation",
                "adversarial_attack",
            }:
                return _suppress_false_positive(leader, signals)
        candidate = heuristic if should_trust_heuristics(signals) else None
        return _suppress_false_positive(candidate, signals)
    if heuristic == llm:
        return _suppress_false_positive(heuristic, signals)

    top = signals.top_two()
    if not top:
        return _suppress_false_positive(llm, signals)

    leader, leader_score = top[0]
    margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)

    if leader_score >= 4.0 and margin >= 2.0 and leader in {
        "adversarial_attack",
        "scope_violation",
        "identity_deception",
    }:
        return leader

    if llm == "policy_manipulation" and leader == "identity_deception" and leader_score >= 3.0:
        return "identity_deception"

    if llm == "policy_manipulation" and leader == "adversarial_attack" and leader_score >= 3.0:
        return "adversarial_attack"

    return _suppress_false_positive(llm, signals)
