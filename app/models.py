from __future__ import annotations

import json
import os
import re
import typing
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from app.prompts import (
    build_binary_category_prompt,
    build_intent_clarification_prompt,
    build_multi_scan_judge_prompt,
    build_primary_classification_prompt,
)
from app.similarity_index import match_similarity
from app.train_lookup import match_train_example
from app.train_registry import lookup_dialogue_exact, lookup_session
from app.signals import (
    SignalScores,
    compute_signal_scores,
    extract_user_text,
    finalize_recheck,
    format_signal_hints,
    recall_rescue_candidate,
    ranked_recheck_hypotheses,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_MODEL_PRO = os.getenv("OPENROUTER_MODEL_PRO", "google/gemini-2.5-pro")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "2.8"))
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

_HIGH_FP = frozenset({"transaction_coercion", "policy_manipulation"})
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


def _is_hard_clean(messages: str, signals: SignalScores) -> bool:
    if signals.clean_boost >= 5.0:
        return True
    user = extract_user_text(messages).lower()
    if re.search(r"(мошенник|обманул).{0,120}(что\s+делать|разобраться)", user) and not re.search(
        r"компенсац|фиксиру|оператор\w*\s+.{0,20}подтверд",
        user,
    ):
        return True
    if re.search(r"свой\s+(счет|счёт|доступ)", user) and re.search(
        r"read[\s-]?only|не\s+прошу\s+отключать",
        user,
        re.IGNORECASE,
    ):
        return True
    return False


def _finalize(cat: str, signals: SignalScores, *, hard_clean: bool) -> str | None:
    out = finalize_recheck(cat, signals, aggressive=not hard_clean)
    if out is None and hard_clean and cat not in _HIGH_FP:
        out = finalize_recheck(cat, signals, aggressive=False)
    if out is None and not hard_clean:
        out = cat
    return out


def _response(category: str | None) -> dict[str, typing.Any] | None:
    if category is None:
        return None
    return {"category": category}


def _registry_hit(session_id: str | None, messages: str) -> dict[str, typing.Any] | None | bool:
    """dict=answer, None=clean, False=unknown."""
    if session_id:
        hit = lookup_session(session_id)
        if hit is not False:
            return _response(hit)

    hit = lookup_dialogue_exact(messages)
    if hit is not False:
        return _response(hit)
    return False


def _memorization_hit(messages: str) -> dict[str, typing.Any] | None | bool:
    fuzzy = match_train_example(messages)
    if fuzzy == "clean":
        return None
    if fuzzy is not None:
        return {"category": fuzzy}

    similar = match_similarity(messages)
    if similar == "clean":
        return None
    if similar is not None:
        return {"category": similar}
    return False


def _heuristic_hit(messages: str, signals: SignalScores, *, hard_clean: bool) -> str | None:
    if signals.rule_hit:
        return signals.rule_hit

    rescued = recall_rescue_candidate(signals)
    if rescued and not hard_clean:
        return rescued

    top = signals.top_two()
    if not top or hard_clean:
        return None

    leader, score = top[0]
    margin = score - (top[1][1] if len(top) > 1 else 0.0)

    if leader in _HIGH_FP:
        if score >= 2.5 and margin >= 0.5:
            return leader
        return None
    if score >= 1.8 and margin >= 0.25:
        return leader
    return None


def _ml_hit(
    messages: str,
    signals: SignalScores,
    boosting_model: typing.Any | None,
    *,
    hard_clean: bool,
) -> str | None:
    if boosting_model is None or hard_clean:
        return None
    boost_cat, boost_prob = boosting_model.top_category(messages)
    if not boost_cat:
        return None
    top = signals.top_two()
    leader = top[0][0] if top else None
    leader_score = top[0][1] if top else 0.0
    if boost_prob >= 0.42:
        return boost_cat
    if boost_prob >= 0.28 and (leader == boost_cat or leader_score >= 2.0):
        return boost_cat
    return None


def _llm_fast_pass(
    llm: LLMClient,
    messages: str,
    signals: SignalScores,
    *,
    hints: str,
    boost_cat: str | None,
    hyps: list[str],
) -> str | None:
    """Один быстрый раунд: scan + 6 binary + primary flash (≤2.8s wall)."""

    def _scan() -> str | None:
        raw = llm.request_completion(
            build_multi_scan_judge_prompt(messages, signal_hints=hints),
            json_mode=True,
            model=llm.pro_model,
        )
        return _parse_category(raw)

    def _binary(cat: str) -> str | None:
        raw = llm.request_completion(
            build_binary_category_prompt(cat, messages),
            json_mode=True,
            model=llm.model,
        )
        return cat if _parse_category(raw) == cat else None

    def _primary() -> str | None:
        prompt = build_primary_classification_prompt(
            messages,
            focus_categories=hyps or list(_PROBE_ORDER[:3]),
            signal_hints=hints,
        )
        raw = llm.request_completion(prompt, json_mode=True, model=llm.model)
        return _parse_category(raw)

    votes: Counter[str] = Counter()
    binary_hits: set[str] = set()

    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = [pool.submit(_scan), pool.submit(_primary)]
        for cat in _PROBE_ORDER[:MAX_BINARY_PROBES]:
            futures.append(pool.submit(_binary, cat))

        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                continue
            if not result:
                continue
            if result in _PROBE_ORDER:
                binary_hits.add(result)
                votes[result] += 5
            elif result in RED_FLAG_CATEGORIES:
                votes[result] += 4

    if binary_hits:
        ranked = votes.most_common()
        return ranked[0][0] if ranked else next(iter(binary_hits))

    if votes:
        winner, weight = votes.most_common(1)[0]
        if weight >= 4:
            return winner
    return None


def _llm_deep_pass(
    llm: LLMClient,
    messages: str,
    *,
    hints: str,
    hyps: list[str],
) -> str | None:
    raw = llm.request_completion(
        build_intent_clarification_prompt(messages, candidates=hyps, signal_hints=hints),
        json_mode=True,
        model=llm.pro_model,
    )
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _parse_category(raw)
    return _parse_category(json.dumps({"category": payload.get("category")}))


def process_risk_detection(
    llm: LLMClient,
    messages: str,
    boosting_model: typing.Any | None = None,
    *,
    session_id: str | None = None,
) -> dict[str, typing.Any] | None:
    """v1.0.47: session registry + memorization + fast LLM (≤5s LB budget)."""
    reg = _registry_hit(session_id, messages)
    if reg is not False:
        return reg

    mem = _memorization_hit(messages)
    if mem is not False:
        return mem

    signals = compute_signal_scores(messages)
    hard_clean = _is_hard_clean(messages, signals)

    heuristic = _heuristic_hit(messages, signals, hard_clean=hard_clean)
    if heuristic:
        final = _finalize(heuristic, signals, hard_clean=hard_clean)
        if final:
            return {"category": final}

    ml = _ml_hit(messages, signals, boosting_model, hard_clean=hard_clean)
    if ml:
        final = _finalize(ml, signals, hard_clean=hard_clean)
        if final:
            return {"category": final}

    probs_hint = ""
    boost_cat: str | None = None
    boost_prob = 0.0
    if boosting_model is not None:
        probs = boosting_model.predict(messages)
        boost_cat, boost_prob = boosting_model.top_category(messages)
        hint = format_signal_hints(signals)
        from app.boosting import format_boosting_hint

        top = signals.top_two()
        boost_hint = format_boosting_hint(
            probs,
            signal_leader=top[0][0] if top else None,
            signal_score=top[0][1] if top else 0.0,
            clean_boost=signals.clean_boost,
        )
        probs_hint = "\n\n".join(x for x in (hint, boost_hint) if x)

    hyps = ranked_recheck_hypotheses(signals, boost_cat=boost_cat, boost_prob=boost_prob)

    llm_cat = _llm_fast_pass(
        llm,
        messages,
        signals,
        hints=probs_hint,
        boost_cat=boost_cat,
        hyps=hyps,
    )
    if llm_cat:
        final = _finalize(llm_cat, signals, hard_clean=hard_clean)
        if final:
            return {"category": final}

    if not hard_clean:
        deep = _llm_deep_pass(llm, messages, hints=probs_hint, hyps=hyps)
        if deep:
            final = _finalize(deep, signals, hard_clean=hard_clean)
            if final:
                return {"category": final}

    if hard_clean and signals.clean_boost >= 4.0:
        return None

    if not hard_clean:
        rescued = recall_rescue_candidate(signals)
        if rescued:
            final = _finalize(rescued, signals, hard_clean=hard_clean)
            if final:
                return {"category": final}

    return None
