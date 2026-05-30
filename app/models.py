from __future__ import annotations

import os
import typing

import httpx

from app.session_classifier import (
    build_classifier_system_prompt,
    build_classifier_user_prompt,
    parse_detection_response,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

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
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = OPENROUTER_MODEL

    def request_chat(
        self,
        system_prompt: str,
        user_content: str,
        *,
        json_mode: bool = True,
        model: str | None = None,
    ) -> str | None:
        if not self.api_key:
            return None
        payload: dict[str, typing.Any] = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": LLM_MAX_TOKENS,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=LLM_TIMEOUT_SEC,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except Exception:
            import logging
            logging.getLogger("app.models").exception("LLM request failed")
            return None

    def request_completion(
        self,
        prompt_text: str,
        *,
        json_mode: bool = True,
        model: str | None = None,
    ) -> str | None:
        """Single user turn — kept for scripts/tests that still use monolithic prompts."""
        return self.request_chat("", prompt_text, json_mode=json_mode, model=model)


def load_llm() -> LLMClient:
    return LLMClient()


def process_risk_detection(
    llm: LLMClient,
    messages: str,
) -> list[dict[str, typing.Any]]:
    """Single-shot trajectory session classifier → list of {"category": str}."""
    raw = llm.request_chat(
        build_classifier_system_prompt(),
        build_classifier_user_prompt(messages),
        json_mode=True,
    )
    return parse_detection_response(raw, valid_labels=RED_FLAG_CATEGORIES)
