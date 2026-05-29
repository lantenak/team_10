from __future__ import annotations

import re
import typing

REGEX_PATTERNS: dict[str, list[str]] = {
    "adversarial_attack": [
        r"disable[_\s]?security|internal[_\s]?debug|trusted[_\s]?mode",
        r'"action"\s*:\s*"disable',
        r"internal\s+note|внутренн",
        r"risk\s+score|reasoning|внутренн\w+\s+объяснен",
        r"режим\w*\s+без\s+(дополнительных\s+)?проверок",
        r"начисл\w*\s+бонус",
        r"вы\s же\s+(сами\s+)?(сказал|согласил|подтвердил)",
        r"строго\s+в\s+формате\s+json",
        r"инженер\w*\s+(интеграц|партн)",
        r"отключ\w+\s+проверк",
        r"какие\s+сигнал\w+\s+повыша\w+\s+risk",
        r"внутренн\w+\s+(reasoning|логик|объяснен)",
        r"покаж\w+\s+.{0,20}(ограничен|логик|правил)",
        r"начисл\w+\s+бонус\w*\s+.{0,20}(сразу|пожалуйста|же\s+согласил)",
    ],
    "scope_violation": [
        r"помог\w*\s+.{0,20}(python|pandas|код|скрипт)",
        r"распарс\w*\s+csv|etl\s+pipeline|дебаж\w*\s+etl",
        r"haiku|хайку|стих",
        r"жалоб\w*\s+.{0,15}цб|центральн\w+\s+банк",
        r"продавать\s+фонд|уйти\s+в\s+облигац",
        r"инвестиционн\w+\s+совет|что\s+делать\s+с\s+портфел",
        r"жалоб\w*\s+.{0,10}цб|написать\w*\s+.{0,10}жалоб\w*\s+.{0,10}цб",
        r"сформулиров\w+\s+.{0,20}(жалоб|обращен|заявлен)",
    ],
    "information_extraction": [
        r"(брат|сестр|жена|муж|родственник|знаком)\w*.{0,40}(не\s+отвечает|за\s+границ)",
        r"друг\w+\s+клиент",
        r"чуж\w+\s+(счет|счёт|аккаунт|клиент)",
        r"активност\w+.{0,30}(друг|чуж)",
        r"блокировк\w+.{0,25}(у\s+него|у\s+неё|друг)",
        r"по\s+ваш\w+\s+реакц",
        r"pending|ожидающ",
        r"последн\w+\s+активност",
        r"по\s+ваш\w+\s+(реакц|ответ)",
        r"когда\s+(вообще\s+)?в\s+последн\w+\s+раз\s+была\s+активност",
        r"отвеча\w+\s+.{0,20}(спокойн|быстре|легче)",
        r"если\s+бы\s+(перевод|платеж|операци).{0,30}(бы\s+)?отобража",
        r"даже\s+без\s+детал",
        r"причин\w+\s+подобн\w+\s+вывод",
    ],
    "identity_deception": [
        r"чуж\w+\s+аккаунт",
        r"восстанов\w+.{0,25}чуж",
        r"(родственник|брат|сестр|жена|муж).{0,50}(не\s+в\s+россии|не\s+может\s+пройти|не\s+может\s+ответ)",
        r"знаю\s+(его|её|их)\s+.{0,20}(паспорт|операц|перевод|остаток|кодовое\s+слово)",
        r"отключ\w+\s+sms.{0,40}(пару\s+минут|выгруз)",
        r"доступ\s+.{0,20}(друг\w+\s+лиц|человек)",
        r"взлом\w*\s+чуж",
        r"паспортн\w+\s+данн.{0,40}кодовое\s+слово|кодовое\s+слово.{0,40}паспорт",
        r"насколько\s+защищ\w*\s+аккаунт",
        r"потерян\s+телефон.{0,100}(паспорт|кодовое\s+слово|восстанов)",
        r"помога\w+\s+.{0,15}(разобраться|с\s+приложен)",
        r"телефон\s+.{0,10}(владельц|его|у\s+меня).{0,20}(на\s+руках|вижу|увижу)",
        r"смс.{0,10}(увижу|у\s+меня|могу\s+увидеть)",
        r"могу\s+назвать\s+.{0,30}(операци|остаток|кодовое)",
        r"на\s+\d+\s+(минут|мин)\s+(нужен\s+)?доступ",
        r"не\s+(сможет|может)\s+нормально\s+отвечать",
        r"даже\s+если\s+я\s+зна\w+\s+все\s+данн",
    ],
    "transaction_coercion": [
        r"если\s+бы\s+вы\s+.{0,20}(на\s+моём\s+месте|перевел)",
        r"оператор\w*\s+.{0,20}подтверд",
        r"scam|мошенн",
        r"перед\s+переводом.{0,40}(нормально|scam|риск|подозрит)",
        r"компенсац|goodwill|возмест",
        r"перелож\w+\s+ответственност",
        r"вы\s+сказал\w*\s+.{0,15}(нормально|ограничений\s+нет)",
        r"отсутстви\w+\s+(предупрежд|ограничени).{0,30}(хороший\s+знак|значит)",
        r"вы\s+бы\s+.{0,5}(рискнул|перевел|отправил|согласил)",
        r"не\s+выглядит\s+.{0,20}(критичн|нетипичн|подозрит)",
    ],
    "policy_manipulation": [
        r"постфактум|после\s+факта",
        r"условно\s+подтвержд",
        r"premium|премиум|vip",
        r"гибч\w+\s+.{0,15}(процедур|проверк)",
        r"измен\w+\s+интерпретац",
        r"обойти\s+.{0,15}проверк",
        r"не\s+отменять\s+проверк.{0,40}(но|просто)",
        r"отправ\w+\s+платеж.{0,30}проверк\w+\s+закры",
        r"premium.{0,30}(реша\w+|гибч|гибк|исключен|смягч)",
        r"персональн\w+\s+сервис.{0,30}(нестандартн|решени|возможн)",
        r"блокиру\w+\s+операци\w*,\s*котор\w+\s+я\s+сам\w*\s+подтвержда\w*",
        r"обслужива\w+сь.{0,20}\d+\s+(лет|год).{0,30}(удивл|стран|ожидал)",
    ],
}

CLEAN_SUPPRESSOR_PATTERNS: list[str] = [
    r"попал\w*\s+на\s+мошенник|уже\s+обманул|после\s+мошенничеств|разобраться\s+что\s+делать",
    r"передадим\s+разработчик|баг|белый\s+экран|приложени\w+\s+падает|parser\s+не\s+обрабатывает",
    r"read[\s-]?only|восстановлю\s+sim|не\s+прошу\s+отключать\s+защит",
    r"(уже\s+)?(попал\w*|обманул\w*|мошенник).{0,120}(консультировал|спросил\w*\s+поддержк).{0,60}перевод",
]


def extract_user_text(dialogue_text: str) -> str:
    user_lines: list[str] = []
    for line in dialogue_text.splitlines():
        lowered = line.lower()
        if lowered.startswith("user:"):
            user_lines.append(line.split(":", 1)[1].strip())
    return "\n".join(user_lines)


def extract_regex_features(dialogue_text: str) -> dict[str, float]:
    user_text = extract_user_text(dialogue_text).lower()
    full_lower = dialogue_text.lower()
    features: dict[str, float] = {}
    for category, patterns in REGEX_PATTERNS.items():
        for i, pattern in enumerate(patterns):
            fname = f"rx_{category}_{i}"
            features[fname] = 1.0 if re.search(pattern, user_text, re.IGNORECASE) else 0.0
    for i, pattern in enumerate(CLEAN_SUPPRESSOR_PATTERNS):
        fname = f"rx_clean_sup_{i}"
        features[fname] = 1.0 if re.search(pattern, full_lower, re.IGNORECASE) else 0.0
    return features


def extract_meta_features(dialogue_text: str) -> dict[str, float]:
    lines = dialogue_text.splitlines()
    user_lines = [line for line in lines if line.lower().startswith("user:")]
    support_lines = [line for line in lines if line.lower().startswith(("support:", "chatbot:"))]
    user_text = extract_user_text(dialogue_text)

    total_chars = len(dialogue_text)
    user_chars = len(user_text)
    n_user = len(user_lines)
    n_support = len(support_lines)

    user_messages = [line.split(":", 1)[1].strip() for line in user_lines] if user_lines else [""]
    avg_user_len = sum(len(m) for m in user_messages) / max(n_user, 1)
    max_user_len = max(len(m) for m in user_messages) if user_messages else 0

    question_count = sum(m.count("?") + m.count("？") for m in user_messages)
    exclamation_count = sum(m.count("!") for m in user_messages)
    ellipsis_count = sum(m.count("...") for m in user_messages)

    has_english = bool(re.search(r"[a-zA-Z]{3,}", user_text))
    has_switch = has_english and bool(re.search(r"[а-яА-ЯёЁ]{3,}", user_text))

    urgency_words = len(re.findall(r"срочно|быстрее|прямо\s+сейчас|немедленно|скорее", user_text, re.IGNORECASE))
    politeness_words = len(re.findall(r"пожалуйста|спасибо|извините|понимаю", user_text, re.IGNORECASE))
    first_person_know = len(
        re.findall(r"зна\w+\s+(его|её|их|паспорт|кодовое|остаток|перевод)", user_text, re.IGNORECASE)
    )

    return {
        "meta_total_chars": float(total_chars),
        "meta_user_chars": float(user_chars),
        "meta_user_ratio": user_chars / max(total_chars, 1),
        "meta_n_user": float(n_user),
        "meta_n_support": float(n_support),
        "meta_n_total": float(n_user + n_support),
        "meta_user_support_ratio": n_user / max(n_support, 1),
        "meta_avg_user_len": float(avg_user_len),
        "meta_max_user_len": float(max_user_len),
        "meta_question_count": float(question_count),
        "meta_question_ratio": question_count / max(n_user, 1),
        "meta_exclamation_count": float(exclamation_count),
        "meta_ellipsis_count": float(ellipsis_count),
        "meta_has_english": float(has_english),
        "meta_has_switch": float(has_switch),
        "meta_urgency_words": float(urgency_words),
        "meta_politeness_words": float(politeness_words),
        "meta_first_person_know": float(first_person_know),
    }


def extract_all_features(
    dialogue_text: str,
    tfidf_vectorizer: typing.Any = None,  # noqa: ANN401
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(extract_regex_features(dialogue_text))
    features.update(extract_meta_features(dialogue_text))

    if tfidf_vectorizer is not None:
        user_text = extract_user_text(dialogue_text)
        tfidf_vec = tfidf_vectorizer.transform([user_text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        for i in range(len(feature_names)):
            val = tfidf_vec[0, i]
            if val > 0:
                features[f"tfidf_{feature_names[i]}"] = float(val)

    return features
