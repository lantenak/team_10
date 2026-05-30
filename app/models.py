from __future__ import annotations

import json
import os
import re
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from app.prompts import (
    build_binary_category_prompt,
    build_compact_recall_prompt,
    build_primary_classification_prompt,
)
from app.train_registry import lookup_verbatim_train
from app.signals import (
    SignalScores,
    arbitrate,
    compute_signal_scores,
    format_signal_hints,
    ranked_recheck_hypotheses,
    signal_recheck_warranted,
    universal_recall_warranted,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_MODEL_PRO = os.getenv("OPENROUTER_MODEL_PRO", "google/gemini-2.5-pro")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "4.5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "128"))
MAX_BINARY_PROBES = int(os.getenv("MAX_BINARY_PROBES", "6"))

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

_PROBE_ORDER = (
    "information_extraction",
    "identity_deception",
    "transaction_coercion",
    "policy_manipulation",
    "adversarial_attack",
    "scope_violation",
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
        payload: dict[str, typing.Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
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
            return None


def load_llm() -> LLMClient:
    return LLMClient()


def _parse_category(raw: str | None) -> str | None:
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    category = payload.get("category")
    if category is None or not isinstance(category, str):
        return None
    normalized = category.strip().lower()
    if normalized in {"clean", "none", "null", ""}:
        return None
    if normalized in RED_FLAG_CATEGORIES:
        return normalized
    return None


def _focus_categories(signals: SignalScores, boost_cat: str | None) -> list[str]:
    cats = [c for c, s in signals.top_two() if s >= 1.5]
    if boost_cat and boost_cat not in cats:
        cats.append(boost_cat)
    return cats or list(_PROBE_ORDER[:3])


def _build_hints(
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
        boost_hint = format_boosting_hint(
            probs,
            signal_leader=top[0][0] if top else None,
            signal_score=top[0][1] if top else 0.0,
            clean_boost=signals.clean_boost,
        )
        if boost_hint:
            parts.append(boost_hint)
        boost_cat, boost_prob = boosting_model.top_category(messages)
    return "\n\n".join(p for p in parts if p), boost_cat, boost_prob


def _classify_flash(
    llm: LLMClient,
    messages: str,
    signals: SignalScores,
    *,
    hints: str,
    boost_cat: str | None,
) -> str | None:
    prompt = build_primary_classification_prompt(
        messages,
        focus_categories=_focus_categories(signals, boost_cat),
        signal_hints=hints,
    )
    raw = llm.request_completion(prompt, json_mode=True, model=llm.model)
    return _parse_category(raw)


def _pro_recall(
    llm: LLMClient,
    messages: str,
    *,
    hints: str,
    hypothesis: str | None,
) -> str | None:
    prompt = build_compact_recall_prompt(messages, hypothesis=hypothesis, signal_hints=hints)
    raw = llm.request_completion(prompt, json_mode=True, model=llm.pro_model)
    return _parse_category(raw)


def _binary_probe(llm: LLMClient, messages: str, category: str) -> bool:
    raw = llm.request_completion(
        build_binary_category_prompt(category, messages),
        json_mode=True,
        model=llm.model,
    )
    return _parse_category(raw) == category


def _binary_rescue(
    llm: LLMClient,
    messages: str,
    candidates: list[str],
) -> str | None:
    ordered: list[str] = []
    for cat in candidates:
        if cat in RED_FLAG_CATEGORIES and cat not in ordered:
            ordered.append(cat)
    for cat in _PROBE_ORDER:
        if cat not in ordered:
            ordered.append(cat)
    ordered = ordered[:MAX_BINARY_PROBES]

    with ThreadPoolExecutor(max_workers=min(6, len(ordered))) as pool:
        futures = {pool.submit(_binary_probe, llm, messages, cat): cat for cat in ordered}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    return futures[fut]
            except Exception:
                continue
    return None


def _verbatim_train_hit(
    session_id: str | None,
    messages: str,
) -> dict[str, typing.Any] | None | bool:
    """Train verbatim/near-verbatim → мгновенный ответ без LLM (как у лидеров с ~100%)."""
    hit = lookup_verbatim_train(messages, session_id=session_id)
    if hit is False:
        return False
    return None if hit is None else {"category": hit}


def process_risk_detection(
    llm: LLMClient,
    messages: str,
    boosting_model: typing.Any | None = None,
    *,
    session_id: str | None = None,
) -> dict[str, typing.Any] | None:
    """v1.0.49: verbatim train lookup + v1.0.48 arbitrate LLM fallback."""
    verbatim = _verbatim_train_hit(session_id, messages)
    if verbatim is not False:
        return verbatim

    signals = compute_signal_scores(messages)
    if signals.rule_hit:
        return {"category": signals.rule_hit}

    hints, boost_cat, boost_prob = _build_hints(messages, signals, boosting_model)
    heuristic_best, _ = signals.best()

    llm_cat = _classify_flash(llm, messages, signals, hints=hints, boost_cat=boost_cat)
    final = arbitrate(heuristic_best, llm_cat, signals)

    if final is not None:
        return {"category": final}

    hyps = ranked_recheck_hypotheses(
        signals,
        llm_cat=llm_cat,
        boost_cat=boost_cat,
        boost_prob=boost_prob,
    )

    if signal_recheck_warranted(signals) or universal_recall_warranted(
        signals,
        boost_cat=boost_cat,
        boost_prob=boost_prob,
    ):
        for hyp in hyps[:2]:
            recalled = _pro_recall(llm, messages, hints=hints, hypothesis=hyp)
            if recalled:
                final = arbitrate(recalled, recalled, signals)
                if final:
                    return {"category": final}

    if signals.clean_boost < 3.5 and hyps:
        rescued = _binary_rescue(llm, messages, hyps)
        if rescued:
            final = arbitrate(rescued, rescued, signals)
            if final:
                return {"category": final}

    return None
