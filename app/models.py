"""LLM-клиент и детектор red flags."""

from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompts import build_classification_prompt
from app.signals import arbitrate, compute_signal_scores, should_trust_heuristics

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")
# Лимит лидерборда: avg ≤ 5000 ms на /check — запас на сеть и парсинг
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "120"))

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

    def request_completion(self, prompt_text: str, *, json_mode: bool = True) -> str | None:
        if not self.api_key:
            return None

        request_payload: dict[str, typing.Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
            "max_tokens": LLM_MAX_TOKENS,
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
                timeout=LLM_TIMEOUT_SEC,
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


def _classify_with_llm(llm_client: LLMClient, messages: str) -> str | None:
    prompt = build_classification_prompt(messages)
    raw = llm_client.request_completion(prompt, json_mode=True)
    return _parse_category(raw)


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
) -> dict[str, typing.Any] | None:
    """Гибрид: эвристики по намерению + LLM + арбитраж. None = clean."""
    signals = compute_signal_scores(messages)

    if signals.rule_hit:
        return {"category": signals.rule_hit}

    heuristic_best, _ = signals.best()

    if should_trust_heuristics(signals) and heuristic_best:
        return {"category": heuristic_best}

    llm_category = _classify_with_llm(llm_client, messages)
    final = arbitrate(heuristic_best, llm_category, signals)

    if final is None:
        return None
    return {"category": final}
