"""LLM-клиент и гибридный детектор red flags (v1.0.3)."""

from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompts import build_classification_prompt
from app.signals import (
    arbitrate,
    compute_signal_scores,
    format_signal_hints,
    is_gray_zone,
    is_likely_clean,
    should_trust_heuristics,
)

OPENROUTER_MODEL_FLASH = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_MODEL_PRO = os.getenv("OPENROUTER_MODEL_PRO", "google/gemini-2.5-pro")
LLM_TIMEOUT_FLASH_SEC = 4.0
LLM_TIMEOUT_PRO_SEC = 4.5

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

        chosen_model = model or OPENROUTER_MODEL_FLASH
        timeout = timeout_sec or (LLM_TIMEOUT_PRO_SEC if chosen_model == OPENROUTER_MODEL_PRO else LLM_TIMEOUT_FLASH_SEC)

        request_payload: dict[str, typing.Any] = {
            "model": chosen_model,
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
                timeout=timeout,
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
    use_pro: bool,
    signal_hints: str,
) -> str | None:
    prompt = build_classification_prompt(messages, signal_hints=signal_hints)
    model = OPENROUTER_MODEL_PRO if use_pro else OPENROUTER_MODEL_FLASH
    raw = llm_client.request_completion(prompt, model=model, json_mode=True)
    return _parse_category(raw)


def _gray_zone_rescue(signals) -> str | None:  # noqa: ANN001
    """Только для серой зоны: если LLM/clean, но сигнал сильный — не пропустить."""
    if signals.clean_boost >= 3.0:
        return None
    ranked = signals.top_two()
    if not ranked:
        return None
    leader, leader_score = ranked[0]
    margin = leader_score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if leader_score >= 3.5 and margin >= 1.5:
        return leader
    return None


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
) -> dict[str, typing.Any] | None:
    """1.0.3: быстрые пути + Flash + Pro в серой зоне + точечный recall-rescue."""
    signals = compute_signal_scores(messages)

    if signals.rule_hit:
        return {"category": signals.rule_hit}

    if is_likely_clean(signals):
        return None

    heuristic_best, _ = signals.best()

    if should_trust_heuristics(signals) and heuristic_best:
        return {"category": heuristic_best}

    gray = is_gray_zone(signals)
    hints = format_signal_hints(signals) if gray else ""

    llm_category = _classify_with_llm(
        llm_client,
        messages,
        use_pro=False,
        signal_hints=hints,
    )
    final = arbitrate(heuristic_best, llm_category, signals)

    if final is None and gray:
        rescued = _gray_zone_rescue(signals)
        if rescued:
            final = rescued

    if final is None:
        return None
    return {"category": final}
