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
    build_pairwise_prompt,
    build_rescue_classification_prompt,
)
from app.signals import (
    PAIRWISE_MARGIN,
    SignalScores,
    arbitrate,
    compute_signal_scores,
    format_signal_hints,
    get_pairwise_disambiguation,
    is_strong_clean_context,
    needs_pro_gray_rescue,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_RESCUE_MODEL = os.getenv("OPENROUTER_RESCUE_MODEL", "google/gemini-2.5-pro")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.5"))
RESCUE_TIMEOUT_SEC = float(os.getenv("RESCUE_TIMEOUT_SEC", "5.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "120"))
USE_PRO_RESCUE = os.getenv("USE_PRO_RESCUE", "true").lower() in ("1", "true", "yes")
USE_PAIRWISE = os.getenv("USE_PAIRWISE", "true").lower() in ("1", "true", "yes")

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

    def __init__(
        self,
        model: str | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or OPENROUTER_MODEL
        self.timeout_sec = timeout_sec if timeout_sec is not None else LLM_TIMEOUT_SEC

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
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001
            return None


@typing.final
class HybridLLMClient:
    """Flash — основной проход; Pro — серой зоны (recall) и pairwise."""

    def __init__(self) -> None:
        self.primary = LLMClient(OPENROUTER_MODEL, LLM_TIMEOUT_SEC)
        self.rescue = LLMClient(OPENROUTER_RESCUE_MODEL, RESCUE_TIMEOUT_SEC)

    @property
    def api_key(self) -> str:
        return self.primary.api_key

    @property
    def model(self) -> str:
        return f"{self.primary.model}+{self.rescue.model}"


def load_llm() -> HybridLLMClient:
    return HybridLLMClient()


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


def _request_category(client: LLMClient, prompt_text: str) -> tuple[str | None, bool]:
    raw = client.request_completion(prompt_text, json_mode=True)
    if raw is None:
        return None, False
    return _parse_category(raw), True


def _classify_with_llm(
    client: LLMClient,
    messages: str,
    signals: SignalScores,
) -> tuple[str | None, bool]:
    hints = format_signal_hints(signals)
    prompt = build_classification_prompt(messages, signal_hints=hints)
    return _request_category(client, prompt)


def _classify_compact(client: LLMClient, messages: str, signals: SignalScores) -> str | None:
    hints = format_signal_hints(signals)
    prompt = build_compact_classification_prompt(messages, signal_hints=hints)
    category, _ = _request_category(client, prompt)
    return category


def _classify_pro_rescue(
    client: LLMClient,
    messages: str,
    signals: SignalScores,
) -> str | None:
    top = signals.top_two()
    if not top:
        return None
    suspected = top[0][0]
    hints = format_signal_hints(signals)
    prompt = build_rescue_classification_prompt(
        messages,
        suspected_category=suspected,
        signal_hints=hints,
    )
    category, _ = _request_category(client, prompt)
    return category


def _classify_pairwise(
    client: LLMClient,
    messages: str,
    pair: tuple[str, str],
    signals: SignalScores,
) -> str | None:
    hints = format_signal_hints(signals)
    prompt = build_pairwise_prompt(
        messages,
        category_a=pair[0],
        category_b=pair[1],
        signal_hints=hints,
    )
    category, _ = _request_category(client, prompt)
    if category in pair:
        return category
    return None


def process_risk_detection(
    llm_client: HybridLLMClient,
    messages: str,
) -> dict[str, typing.Any] | None:
    """v1.0.25 пайплайн + Pro в серой зоне + pairwise на путанице."""
    signals = compute_signal_scores(messages)
    heuristic_best, _ = signals.best()

    if signals.rule_hit:
        return {"category": signals.rule_hit}

    llm_category, api_ok = _classify_with_llm(llm_client.primary, messages, signals)

    if not api_ok and not is_strong_clean_context(signals):
        llm_category = _classify_compact(llm_client.primary, messages, signals)

    final = arbitrate(heuristic_best, llm_category, signals)

    if final is None and USE_PRO_RESCUE and needs_pro_gray_rescue(signals):
        rescue = _classify_pro_rescue(llm_client.rescue, messages, signals)
        if rescue is not None:
            final = arbitrate(heuristic_best, rescue, signals)

    if USE_PAIRWISE:
        pair = get_pairwise_disambiguation(signals, final)
        if pair is not None:
            top = signals.top_two()
            margin = 999.0
            if top and len(top) > 1:
                margin = top[0][1] - top[1][1]
            if final is None or margin < PAIRWISE_MARGIN:
                pairwise = _classify_pairwise(llm_client.rescue, messages, pair, signals)
                if pairwise is not None:
                    final = pairwise

    if final is None:
        return None
    return {"category": final}
