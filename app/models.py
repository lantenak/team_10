from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompts import build_classification_prompt, build_compact_recall_prompt
from app.signals import (
    SignalScores,
    arbitrate,
    compute_signal_scores,
    finalize_recheck,
    format_signal_hints,
    recall_rescue_candidate,
    signal_recheck_warranted,
    universal_recall_warranted,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_MODEL_PRO = os.getenv("OPENROUTER_MODEL_PRO", "google/gemini-2.5-pro")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "120"))
USE_BOOST_RECHECK = os.getenv("USE_BOOST_RECHECK", "true").lower() in ("1", "true", "yes")
USE_PRO_RECALL = os.getenv("USE_PRO_RECALL", "true").lower() in ("1", "true", "yes")

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
        self.pro_model = OPENROUTER_MODEL_PRO

    def request_completion(
        self,
        prompt_text: str,
        *,
        json_mode: bool = True,
        model: str | None = None,
    ) -> str | None:
        if not self.api_key:
            return None

        request_payload: dict[str, typing.Any] = {
            "model": model or self.model,
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


def _recall_model(llm_client: LLMClient) -> str:
    return llm_client.pro_model if USE_PRO_RECALL else llm_client.model


def _run_compact_recall(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    hint_block: str,
    *,
    hypothesis: str | None = None,
) -> str | None:
    prompt = build_compact_recall_prompt(
        messages,
        hypothesis=hypothesis,
        signal_hints=hint_block,
    )
    raw = llm_client.request_completion(
        prompt,
        json_mode=True,
        model=_recall_model(llm_client),
    )
    return finalize_recheck(_parse_category(raw), signals)


def _maybe_signal_recheck(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    hint_block: str,
) -> str | None:
    warrant = signal_recheck_warranted(signals)
    if warrant is None:
        return None
    leader, _ = warrant
    return _run_compact_recall(
        llm_client,
        messages,
        signals,
        hint_block,
        hypothesis=leader,
    )


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


def _maybe_boost_recheck(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    hint_block: str,
    boost_cat: str | None,
    boost_prob: float,
) -> str | None:
    if not USE_BOOST_RECHECK or boost_cat is None:
        return None
    if signals.clean_boost >= 2.5:
        return None

    top = signals.top_two()
    leader, leader_score = top[0] if top else (None, 0.0)
    margin = leader_score - (top[1][1] if top and len(top) > 1 else 0.0)

    signals_agree = leader == boost_cat and leader_score >= 2.0 and margin >= 0.6
    if signals_agree and boost_prob >= 0.35:
        hypothesis = boost_cat
    elif boost_prob >= 0.55 and signals.clean_boost < 1.5:
        hypothesis = boost_cat
    else:
        return None

    return _run_compact_recall(
        llm_client,
        messages,
        signals,
        hint_block,
        hypothesis=hypothesis,
    )


def _maybe_universal_pro_recall(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    hint_block: str,
    boost_cat: str | None,
    boost_prob: float,
) -> str | None:
    if not USE_PRO_RECALL:
        return None
    if not universal_recall_warranted(signals, boost_cat=boost_cat, boost_prob=boost_prob):
        return None

    top = signals.top_two()
    hypothesis = top[0][0] if top and top[0][1] >= 2.0 else boost_cat
    return _run_compact_recall(
        llm_client,
        messages,
        signals,
        hint_block,
        hypothesis=hypothesis,
    )


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
    boosting_model: typing.Any | None = None,
) -> dict[str, typing.Any] | None:
    """v1.0.41: Flash primary + Pro compact recall cascade."""
    signals = compute_signal_scores(messages)
    heuristic_best, _ = signals.best()

    if signals.rule_hit:
        return {"category": signals.rule_hit}

    hint_block, boost_cat, boost_prob = _build_hint_block(messages, signals, boosting_model)
    llm_cat = _classify_with_llm(llm_client, messages, hint_block=hint_block)
    final = arbitrate(heuristic_best, llm_cat, signals)

    if final is None:
        final = _maybe_signal_recheck(llm_client, messages, signals, hint_block)

    if final is None:
        final = _maybe_boost_recheck(
            llm_client,
            messages,
            signals,
            hint_block,
            boost_cat,
            boost_prob,
        )

    if final is None:
        final = _maybe_universal_pro_recall(
            llm_client,
            messages,
            signals,
            hint_block,
            boost_cat,
            boost_prob,
        )

    if final is None:
        final = recall_rescue_candidate(signals)

    if final is None:
        return None
    return {"category": final}
