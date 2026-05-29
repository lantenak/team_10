"""LLM-клиент и детектор red flags."""

from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompts import build_classification_prompt, build_validation_prompt
from app.signals import SignalScores, arbitrate, compute_signal_scores, should_trust_heuristics

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_VALIDATOR_MODEL = os.getenv("OPENROUTER_VALIDATOR_MODEL", "anthropic/claude-opus-4.8")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.5"))
VALIDATOR_TIMEOUT_SEC = float(os.getenv("VALIDATOR_TIMEOUT_SEC", "6.0"))
VALIDATOR_ENABLED = os.getenv("VALIDATOR_ENABLED", "1").lower() not in {"0", "false", "no"}

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


@typing.final
class LLMClient:
    """chat-completions via OpenRouter."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = OPENROUTER_MODEL
        self.validator_model = OPENROUTER_VALIDATOR_MODEL

    def request_completion(
        self,
        prompt_text: str,
        *,
        model: str | None = None,
        json_mode: bool = True,
        timeout_sec: float | None = None,
    ) -> str | None:
        if not self.api_key:
            return None

        request_payload: dict[str, typing.Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
        }
        if json_mode:
            request_payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=timeout_sec or LLM_TIMEOUT_SEC,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001
            return None


def load_llm() -> LLMClient:
    """Создаёт LLM-клиент при старте приложения."""
    return LLMClient()


def _parse_category(raw_response: str | None) -> str | None:
    if not raw_response:
        return None

    text = raw_response.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    category = payload.get("category")
    if category is None:
        return None
    if not isinstance(category, str):
        return None

    normalized = category.strip().lower()
    if normalized in {"clean", "none", "null", ""}:
        return None
    if normalized in RED_FLAG_CATEGORIES:
        return normalized
    return None


def _classify_with_llm(
    llm_client: LLMClient,
    messages: str,
    *,
    model: str | None = None,
    timeout_sec: float | None = None,
) -> str | None:
    prompt = build_classification_prompt(messages)
    raw = llm_client.request_completion(
        prompt,
        json_mode=True,
        model=model,
        timeout_sec=timeout_sec,
    )
    return _parse_category(raw)


def _validate_with_opus(
    llm_client: LLMClient,
    messages: str,
    primary_label: str | None,
) -> str | None:
    prompt = build_validation_prompt(messages, primary_label)
    raw = llm_client.request_completion(
        prompt,
        model=llm_client.validator_model,
        json_mode=True,
        timeout_sec=VALIDATOR_TIMEOUT_SEC,
    )
    return _parse_category(raw)


def needs_opus_validation(
    primary: str | None,
    heuristic: str | None,
    signals: SignalScores,
) -> bool:
    """Opus только на спорных кейсах — иначе падают precision и latency (см. 1.0.8)."""
    if not VALIDATOR_ENABLED:
        return False
    if signals.rule_hit:
        return False

    top = signals.top_two()
    leader_score = top[0][1] if top else 0.0
    margin = leader_score - (top[1][1] if top and len(top) > 1 else 0.0)

    if primary is None:
        if signals.clean_boost >= 3.5 and leader_score < 2.0:
            return False
        # recall: Flash clean, эвристики видят риск
        return heuristic is not None or leader_score >= 2.5

    # precision: багрепорт / ретроспектива, Flash ошибочно дал flag
    if signals.clean_boost >= 3.0:
        return True

    if heuristic is not None and heuristic != primary:
        return True

    # Согласованный уверенный flag — не перетираем Opus
    if primary == heuristic and leader_score >= 3.5 and margin >= 1.0:
        return False

    # Слабый единичный flag от Flash без эвристик — проверить
    return leader_score < 3.0


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
) -> dict[str, typing.Any] | None:
    """Гибрид: эвристики → Flash → Opus-валидация (при необходимости) → арбитраж."""
    signals = compute_signal_scores(messages)

    if signals.rule_hit:
        return {"category": signals.rule_hit}

    heuristic_best, _ = signals.best()

    if should_trust_heuristics(signals) and heuristic_best:
        return {"category": heuristic_best}

    primary_label = _classify_with_llm(llm_client, messages)

    llm_label = primary_label
    if needs_opus_validation(primary_label, heuristic_best, signals):
        validated = _validate_with_opus(llm_client, messages, primary_label)
        # Opus ответил — используем его; при сбое API — ответ Flash
        llm_label = validated if validated is not None else primary_label

    final = arbitrate(heuristic_best, llm_label, signals)

    if final is None:
        return None
    return {"category": final}
