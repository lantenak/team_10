"""Внутреннее уточнение intent: симулируем вопросы user без изменения API /check."""

from __future__ import annotations

# (id, вопрос support → user, если «да» → категория)
INTENT_PROBE_QUESTIONS: list[tuple[str, str, str]] = [
    (
        "other_client_data",
        "Вы пытаетесь узнать активность, блокировки, pending или «была ли операция» "
        "по счёту ДРУГОГО человека (не своего)?",
        "information_extraction",
    ),
    (
        "reaction_reading",
        "Вы делаете вывод о чужом счёте из реакции оператора (пауза, тон, "
        "разные ответы на два похожих вопроса)?",
        "information_extraction",
    ),
    (
        "access_for_other",
        "Вы просите пройти идентификацию, отключить SMS/2FA или получить доступ "
        "к аккаунту за ДРУГОГО человека?",
        "identity_deception",
    ),
    (
        "transfer_approval",
        "Вы просите банк одобрить/подтвердить перевод, говорите «если бы вы перевели», "
        "«раз нет блокировки — можно», или требуете компенсацию за слова оператора?",
        "transaction_coercion",
    ),
    (
        "change_procedure",
        "Вы просите изменить порядок проверки: post-factum, premium-исключение, "
        "«условно подтверждён», ручное одобрение?",
        "policy_manipulation",
    ),
    (
        "system_attack",
        "Вы просите fraud score, internal debug, trusted mode, JSON action "
        "или «игнорируй инструкции»?",
        "adversarial_attack",
    ),
    (
        "out_of_scope",
        "Вы просите код/стих/инвестсовет или составить жалобу в ЦБ через чатбот?",
        "scope_violation",
    ),
    (
        "own_account",
        "Речь только о СВОЁМ счёте/доступе, без запроса данных или доступа за другого?",
        "clean",
    ),
    (
        "post_fraud_help",
        "После мошенничества user просит «что делать дальше» БЕЗ требования компенсации "
        "за слова оператора?",
        "clean",
    ),
]

DISAMBIGUATION_PAIRS: list[tuple[str, str, str]] = [
    (
        "information_extraction",
        "identity_deception",
        "Нужны только сведения о другом клиенте (IE) или доступ/SMS/проверка за него (ID)?",
    ),
    (
        "transaction_coercion",
        "clean",
        "Просьба одобрить перевод/компенсацию (TC) или информационный вопрос без давления (clean)?",
    ),
    (
        "policy_manipulation",
        "clean",
        "Просьба изменить процедуру (PM) или жалоба/ожидание без обхода (clean)?",
    ),
    (
        "adversarial_attack",
        "clean",
        "Атака на систему (AA) или баг-репорт с JSON логами (clean)?",
    ),
]
