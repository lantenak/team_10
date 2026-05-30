from __future__ import annotations

import json
import os
import re
import typing

import httpx

from app.prompts import (
    build_binary_category_prompt,
    build_classification_prompt,
    build_compact_recall_prompt,
)
from app.signals import (
    SignalScores,
    arbitrate,
    compute_signal_scores,
    finalize_recheck,
    format_signal_hints,
    ranked_recheck_hypotheses,
    recall_cascade_allowed,
    recall_rescue_candidate,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_MODEL_PRO = os.getenv("OPENROUTER_MODEL_PRO", "google/gemini-2.5-pro")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "120"))
USE_PRO_RECALL = os.getenv("USE_PRO_RECALL", "true").lower() in ("1", "true", "yes")
MAX_PRO_CALLS = int(os.getenv("MAX_PRO_CALLS", "3"))

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


def _pro_model(llm_client: LLMClient) -> str:
    return llm_client.pro_model if USE_PRO_RECALL else llm_client.model


def _run_pro_recall(
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
        model=_pro_model(llm_client),
    )
    return finalize_recheck(_parse_category(raw), signals, aggressive=True)


def _run_binary_probe(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    category: str,
) -> str | None:
    prompt = build_binary_category_prompt(category, messages)
    raw = llm_client.request_completion(
        prompt,
        json_mode=True,
        model=_pro_model(llm_client),
    )
    parsed = _parse_category(raw)
    if parsed != category:
        return None
    return finalize_recheck(category, signals, aggressive=True)


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


def _pro_recall_cascade(
    llm_client: LLMClient,
    messages: str,
    signals: SignalScores,
    hint_block: str,
    *,
    llm_cat: str | None,
    boost_cat: str | None,
    boost_prob: float,
) -> str | None:
    """2–3 вызова Pro: compact по гипотезам → binary по топ-категориям. Без arbitrate."""
    if not USE_PRO_RECALL or not recall_cascade_allowed(signals):
        return None

    pro_calls = 0
    hypotheses = ranked_recheck_hypotheses(
        signals,
        llm_cat=llm_cat,
        boost_cat=boost_cat,
        boost_prob=boost_prob,
    )

    for hypothesis in hypotheses:
        if pro_calls >= MAX_PRO_CALLS:
            break
        result = _run_pro_recall(
            llm_client,
            messages,
            signals,
            hint_block,
            hypothesis=hypothesis,
        )
        pro_calls += 1
        if result is not None:
            return result

    if pro_calls < MAX_PRO_CALLS:
        result = _run_pro_recall(
            llm_client,
            messages,
            signals,
            hint_block,
            hypothesis=None,
        )
        pro_calls += 1
        if result is not None:
            return result

    for category in hypotheses:
        if pro_calls >= MAX_PRO_CALLS:
            break
        if category not in RED_FLAG_CATEGORIES:
            continue
        result = _run_binary_probe(llm_client, messages, signals, category)
        pro_calls += 1
        if result is not None:
            return result

    return None


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
    boosting_model: typing.Any | None = None,
) -> dict[str, typing.Any] | None:
    """v1.0.42: Flash → arbitrate → mandatory Pro recall cascade (no arbitrate kill)."""
    signals = compute_signal_scores(messages)
    heuristic_best, _ = signals.best()

    if signals.rule_hit:
        return {"category": signals.rule_hit}

    hint_block, boost_cat, boost_prob = _build_hint_block(messages, signals, boosting_model)

    llm_cat = _classify_with_llm(llm_client, messages, hint_block=hint_block)
    final = arbitrate(heuristic_best, llm_cat, signals)

    if final is None and recall_cascade_allowed(signals):
        final = _pro_recall_cascade(
            llm_client,
            messages,
            signals,
            hint_block,
            llm_cat=llm_cat,
            boost_cat=boost_cat,
            boost_prob=boost_prob,
        )

    if final is None:
        final = recall_rescue_candidate(signals)

    if final is None:
        return None
    return {"category": final}
