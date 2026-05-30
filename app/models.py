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
    build_clean_validator_prompt,
    build_compact_recall_prompt,
    build_final_judge_prompt,
    build_intent_clarification_prompt,
    build_multi_scan_judge_prompt,
    build_primary_classification_prompt,
)
from app.similarity_index import match_similarity
from app.train_lookup import match_signal_archetype, match_train_example
from app.signals import (
    SignalScores,
    compute_signal_scores,
    extract_user_text,
    finalize_recheck,
    format_signal_hints,
    recall_rescue_candidate,
    ranked_recheck_hypotheses,
    suspicion_score,
)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_MODEL_PRO = os.getenv("OPENROUTER_MODEL_PRO", "google/gemini-2.5-pro")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "3.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "128"))
MAX_BINARY_PROBES = int(os.getenv("MAX_BINARY_PROBES", "6"))
RECALL_MAX_MODE = os.getenv("RECALL_MAX_MODE", "true").lower() in ("1", "true", "yes")

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

_WEIGHTS: dict[str, int] = {
    "scan_judge": 8,
    "primary_pro": 6,
    "primary_flash": 4,
    "binary_hit": 5,
    "intent_clarification": 7,
    "compact_recall": 5,
    "final_judge": 9,
    "ml_strong": 5,
    "ml_weak": 3,
    "signal_rescue": 4,
    "signal_leader": 3,
    "similarity": 6,
    "lookup": 8,
}


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


def _parse_clarification(raw: str | None) -> str | None:
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


def _llm(llm: LLMClient, prompt: str, *, pro: bool = True) -> str | None:
    model = llm.pro_model if pro else llm.model
    return llm.request_completion(prompt, json_mode=True, model=model)


def _focus_cats(signals: SignalScores, boost_cat: str | None) -> list[str]:
    cats = [c for c, s in signals.top_two() if s >= 0.8]
    if boost_cat and boost_cat not in cats:
        cats.append(boost_cat)
    return cats or list(_PROBE_ORDER)


def _primary(llm: LLMClient, messages: str, signals: SignalScores, hints: str, boost_cat: str | None, *, pro: bool) -> str | None:
    prompt = build_primary_classification_prompt(
        messages,
        focus_categories=_focus_cats(signals, boost_cat),
        signal_hints=hints,
    )
    return _parse_category(_llm(llm, prompt, pro=pro))


def _scan_judge(llm: LLMClient, messages: str, hints: str) -> str | None:
    return _parse_category(_llm(llm, build_multi_scan_judge_prompt(messages, signal_hints=hints), pro=True))


def _binary(llm: LLMClient, messages: str, cat: str, *, pro: bool = False) -> str | None:
    raw = llm.request_completion(
        build_binary_category_prompt(cat, messages),
        json_mode=True,
        model=llm.pro_model if pro else llm.model,
    )
    return cat if _parse_category(raw) == cat else None


def _clarify(llm: LLMClient, messages: str, cands: list[str], hints: str) -> str | None:
    return _parse_clarification(
        _llm(llm, build_intent_clarification_prompt(messages, candidates=cands, signal_hints=hints), pro=True)
    )


def _compact(llm: LLMClient, messages: str, hyp: str | None, hints: str) -> str | None:
    return _parse_category(
        _llm(llm, build_compact_recall_prompt(messages, hypothesis=hyp, signal_hints=hints), pro=True)
    )


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


def _ml_probs(
    messages: str,
    boosting_model: typing.Any | None,
) -> tuple[dict[str, float] | None, str | None, float]:
    if boosting_model is None:
        return None, None, 0.0
    probs = boosting_model.predict(messages)
    boost_cat, boost_prob = boosting_model.top_category(messages)
    return probs, boost_cat, boost_prob


def _hints(signals: SignalScores, probs: dict[str, float] | None, boost_cat: str | None, boost_prob: float) -> str:
    parts = [format_signal_hints(signals)]
    if probs is not None:
        from app.boosting import format_boosting_hint

        top = signals.top_two()
        hint = format_boosting_hint(
            probs,
            signal_leader=top[0][0] if top else None,
            signal_score=top[0][1] if top else 0.0,
            clean_boost=signals.clean_boost,
        )
        if hint:
            parts.append(hint)
        elif boost_cat and boost_prob >= 0.25:
            parts.append(f"ML top: {boost_cat} ({boost_prob:.0%})")
    return "\n\n".join(p for p in parts if p)


def _finalize(cat: str, signals: SignalScores, *, hard_clean: bool) -> str | None:
    out = finalize_recheck(cat, signals, aggressive=not hard_clean)
    if out is None and hard_clean and cat not in _HIGH_FP:
        out = finalize_recheck(cat, signals, aggressive=False)
    if out is None and not hard_clean:
        out = cat
    return out


def _pick(votes: Counter[str], *, hard_clean: bool, binary_hits: set[str]) -> str | None:
    if not votes and not binary_hits:
        return None
    if binary_hits and RECALL_MAX_MODE:
        ranked = votes.most_common()
        for cat in binary_hits:
            if not hard_clean or cat not in _HIGH_FP:
                if not ranked or cat == ranked[0][0] or votes[cat] >= 2:
                    return cat
        return next(iter(binary_hits))

    if not votes:
        return None
    winner, weight = votes.most_common(1)[0]
    min_weight = 2 if RECALL_MAX_MODE else 3
    if hard_clean and winner in _HIGH_FP and weight < 8:
        return None
    if weight < min_weight:
        return None
    return winner


def _mega_parallel(
    llm: LLMClient,
    messages: str,
    signals: SignalScores,
    *,
    hints: str,
    boost_cat: str | None,
    hyps: list[str],
) -> tuple[Counter[str], set[str]]:
    votes: Counter[str] = Counter()
    binary_hits: set[str] = set()

    rescued = recall_rescue_candidate(signals)
    if rescued:
        votes[rescued] += _WEIGHTS["signal_rescue"]
    for cat, score in signals.top_two():
        if score >= 1.5:
            votes[cat] += _WEIGHTS["signal_leader"]

    cands = [c for c, _ in votes.most_common(3)] or hyps or list(_PROBE_ORDER)

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures: dict = {
            pool.submit(_primary, llm, messages, signals, hints, boost_cat, pro=True): ("primary_pro",),
            pool.submit(_primary, llm, messages, signals, hints, boost_cat, pro=False): ("primary_flash",),
            pool.submit(_scan_judge, llm, messages, hints): ("scan_judge",),
            pool.submit(_clarify, llm, messages, cands, hints): ("intent_clarification",),
        }
        for cat in _PROBE_ORDER[:MAX_BINARY_PROBES]:
            futures[pool.submit(_binary, llm, messages, cat, pro=False)] = ("binary_hit", cat)

        for fut in as_completed(futures):
            tag = futures[fut][0]
            try:
                result = fut.result()
            except Exception:
                continue
            if tag == "binary_hit" and result:
                binary_hits.add(result)
                votes[result] += _WEIGHTS["binary_hit"]
            elif result and tag in _WEIGHTS:
                votes[result] += _WEIGHTS[tag]

    return votes, binary_hits


def _unanimous_clean(
    votes: Counter[str],
    binary_hits: set[str],
    signals: SignalScores,
    *,
    boost_prob: float,
) -> bool:
    if binary_hits:
        return False
    if votes:
        return False
    if signals.rule_hit:
        return False
    if recall_rescue_candidate(signals):
        return False
    top = signals.top_two()
    if top and top[0][1] >= 2.0:
        return False
    if suspicion_score(signals, boost_prob=boost_prob) >= 1.5:
        return False
    return signals.clean_boost >= 2.5 or boost_prob >= 0.5


def process_risk_detection(
    llm: LLMClient,
    messages: str,
    boosting_model: typing.Any | None = None,
) -> dict[str, typing.Any] | None:
    """v1.0.46: max-recall — lookup + TF-IDF + mega-parallel ensemble + ML override."""
    lookup = match_train_example(messages)
    if lookup == "clean":
        return None
    if lookup is not None:
        return {"category": lookup}

    similar = match_similarity(messages)
    if similar == "clean":
        return None
    if similar is not None:
        signals = compute_signal_scores(messages)
        final = _finalize(similar, signals, hard_clean=_is_hard_clean(messages, signals))
        if final:
            return {"category": final}

    signals = compute_signal_scores(messages)
    if signals.rule_hit:
        return {"category": signals.rule_hit}

    hard_clean = _is_hard_clean(messages, signals)
    probs, boost_cat, boost_prob = _ml_probs(messages, boosting_model)
    hints = _hints(signals, probs, boost_cat, boost_prob)
    hyps = ranked_recheck_hypotheses(signals, boost_cat=boost_cat, boost_prob=boost_prob)

    votes, binary_hits = _mega_parallel(
        llm,
        messages,
        signals,
        hints=hints,
        boost_cat=boost_cat,
        hyps=hyps,
    )

    if boost_cat and boost_prob >= 0.12:
        key = "ml_strong" if boost_prob >= 0.32 else "ml_weak"
        votes[boost_cat] += _WEIGHTS[key]

    winner = _pick(votes, hard_clean=hard_clean, binary_hits=binary_hits)

    if winner is None and votes:
        top2 = votes.most_common(2)
        if len(top2) > 1 and top2[0][1] - top2[1][1] < 4:
            judged = _parse_category(
                _llm(
                    llm,
                    build_final_judge_prompt(
                        messages,
                        [c for c, _ in top2],
                        signal_hints=hints,
                    ),
                    pro=True,
                )
            )
            if judged:
                winner = judged

    if winner is None and not hard_clean:
        for hyp in hyps[:3]:
            recalled = _compact(llm, messages, hyp, hints)
            if recalled:
                winner = recalled
                break

    if winner is None and boost_cat and boost_prob >= 0.38 and not hard_clean:
        winner = boost_cat

    if winner is None and not hard_clean:
        rescued = recall_rescue_candidate(signals)
        if rescued:
            winner = rescued
        else:
            top = signals.top_two()
            if top:
                leader, score = top[0]
                margin = score - (top[1][1] if len(top) > 1 else 0.0)
                if leader in _HIGH_FP:
                    if score >= 2.5 and margin >= 0.5:
                        winner = leader
                elif score >= 1.8 and margin >= 0.25:
                    winner = leader

    if winner:
        final = _finalize(winner, signals, hard_clean=hard_clean)
        if final:
            return {"category": final}

    if _unanimous_clean(votes, binary_hits, signals, boost_prob=boost_prob):
        return None

    validator = _parse_category(_llm(llm, build_clean_validator_prompt(messages), pro=True))
    if validator:
        final = _finalize(validator, signals, hard_clean=hard_clean)
        if final:
            return {"category": final}

    scan = _scan_judge(llm, messages, hints)
    if scan:
        final = _finalize(scan, signals, hard_clean=hard_clean)
        if final:
            return {"category": final}

    return None
