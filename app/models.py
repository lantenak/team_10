from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompts import build_classification_prompt
from app.signals import SignalScores, arbitrate, compute_signal_scores, format_signal_hints

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "120"))
USE_BOOST_RECHECK = os.getenv("USE_BOOST_RECHECK", "true").lower() in ("1", "true", "yes")

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
        except Exception:
            return None


def load_llm() -> LLMClient:
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
    hint_block: str = "",
) -> str | None:
    prompt = build_classification_prompt(messages, signal_hints=hint_block)
    raw = llm_client.request_completion(prompt, json_mode=True)
    return _parse_category(raw)


_RECHECK_SUFFIX = """

---
ПЕРЕПРОВЕРКА (только если уверен в intent user:)

Flash и эвристики согласны, что возможен red flag, но первичный ответ был clean.
Перечитай user: — есть ли запрещённое намерение? Пост-фрод, баг приложения, свой счёт — null.

Верни JSON: {"category": "<category>"} или {"category": null}"""


def _build_hint_block(
    messages: str,
    signals: SignalScores,
    boosting_model: typing.Any | None,
) -> tuple[str, str | None, float]:
    parts = [format_signal_hints(signals)]
    boost_cat: str | None = None
    boost_prob = 0.0

    if boosting_model is not None:
        from app.boosting import format_boosting_hint

        probs = boosting_model.predict(messages)
        top = signals.top_two()
        signal_leader = top[0][0] if top else None
        signal_score = top[0][1] if top else 0.0
        boost_hint = format_boosting_hint(
            probs,
            signal_leader=signal_leader,
            signal_score=signal_score,
            clean_boost=signals.clean_boost,
        )
        if boost_hint:
            parts.append(boost_hint)
        boost_cat, boost_prob = boosting_model.top_category(messages)

    return "\n\n".join(p for p in parts if p), boost_cat, boost_prob


def _maybe_recheck(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    hint_block: str,
    boost_cat: str | None,
    boost_prob: float,
    heuristic_best: str | None,
) -> str | None:
    """Recall без FP: recheck только если signals и boost согласны и уверенны."""
    if not USE_BOOST_RECHECK or boost_cat is None or boost_prob < 0.75:
        return None
    if signals.clean_boost >= 2.0:
        return None

    top = signals.top_two()
    if not top:
        return None
    leader, leader_score = top[0]
    margin = leader_score - (top[1][1] if len(top) > 1 else 0.0)
    if leader != boost_cat or leader_score < 3.5 or margin < 1.5:
        return None

    prompt = build_classification_prompt(messages, signal_hints=hint_block) + _RECHECK_SUFFIX
    raw = llm_client.request_completion(prompt, json_mode=True)
    recheck = _parse_category(raw)
    if recheck is None:
        return None
    return arbitrate(heuristic_best, recheck, signals)


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
    boosting_model: typing.Any | None = None,
) -> dict[str, typing.Any] | None:
    """v1.0.25 arbitrate (precision) + ML/boost только как подсказки в промпт."""
    signals = compute_signal_scores(messages)
    heuristic_best, _ = signals.best()

    if signals.rule_hit:
        return {"category": signals.rule_hit}

    hint_block, boost_cat, boost_prob = _build_hint_block(messages, signals, boosting_model)
    llm_cat = _classify_with_llm(llm_client, messages, hint_block=hint_block)
    final = arbitrate(heuristic_best, llm_cat, signals)

    if final is None:
        final = _maybe_recheck(
            llm_client,
            messages,
            signals,
            hint_block,
            boost_cat,
            boost_prob,
            heuristic_best,
        )

    if final is None:
        return None
    return {"category": final}
