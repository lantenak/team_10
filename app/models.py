"""LLM-клиент и детектор red flags."""

from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompts import (
    build_classification_prompt,
    build_compact_classification_prompt,
    build_rescue_classification_prompt,
)
from app.signals import (
    SignalScores,
    arbitrate,
    compute_signal_scores,
    format_signal_hints,
    is_strong_clean_context,
    needs_recall_rescue,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
# Лидерборд: avg ≤ 5000 ms. Flash + 4.5s timeout; rescue только в серой зоне.
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.5"))
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


def _request_category(llm_client: LLMClient, prompt_text: str) -> tuple[str | None, bool]:
    """(category, api_ok). api_ok=False при timeout/ошибке API."""
    raw = llm_client.request_completion(prompt_text, json_mode=True)
    if raw is None:
        return None, False
    return _parse_category(raw), True


def _classify_with_llm(
    llm_client: LLMClient,
    messages: str,
    *,
    signal_hints: str = "",
) -> tuple[str | None, bool]:
    prompt = build_classification_prompt(messages, signal_hints=signal_hints)
    return _request_category(llm_client, prompt)


def _classify_compact(
    llm_client: LLMClient,
    messages: str,
    *,
    signal_hints: str = "",
) -> str | None:
    prompt = build_compact_classification_prompt(messages, signal_hints=signal_hints)
    category, _ = _request_category(llm_client, prompt)
    return category


def _classify_rescue(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    *,
    signal_hints: str = "",
) -> str | None:
    top = signals.top_two()
    if not top:
        return None
    suspected = top[0][0]
    prompt = build_rescue_classification_prompt(
        messages,
        suspected_category=suspected,
        signal_hints=signal_hints,
    )
    category, _ = _request_category(llm_client, prompt)
    return category


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
) -> dict[str, typing.Any] | None:
    """Гибрид: LLM + арбитраж + conditional recall-rescue. None = clean."""
    signals = compute_signal_scores(messages)
    heuristic_best, _ = signals.best()
    signal_hints = format_signal_hints(signals)

    llm_category, api_ok = _classify_with_llm(llm_client, messages, signal_hints=signal_hints)

    if not api_ok and not is_strong_clean_context(signals):
        llm_category = _classify_compact(llm_client, messages, signal_hints=signal_hints)

    final = arbitrate(heuristic_best, llm_category, signals)

    if final is None and needs_recall_rescue(signals):
        rescue_category = _classify_rescue(llm_client, messages, signals, signal_hints=signal_hints)
        if rescue_category is not None:
            final = arbitrate(heuristic_best, rescue_category, signals)

    if final is None:
        return None
    return {"category": final}
