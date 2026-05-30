"""Fuzzy lookup + signal-archetype fast path."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.signals import SignalScores, compute_signal_scores
from app.synthetic_fewshots import EXPANDED_FEW_SHOT_EXAMPLES
from app.train_fewshots import TRAIN_FEW_SHOT_EXAMPLES

_ALL_EXAMPLES: list[tuple[str, str]] = TRAIN_FEW_SHOT_EXAMPLES + EXPANDED_FEW_SHOT_EXAMPLES

_MATCH_THRESHOLD = 0.80
_USER_THRESHOLD = 0.82
_JACCARD_THRESHOLD = 0.48

_HIGH_FP = frozenset({"transaction_coercion", "policy_manipulation"})


def _normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s:]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _user_only(dialogue: str) -> str:
    lines = []
    for line in dialogue.splitlines():
        if line.lower().startswith("user:"):
            lines.append(line.split(":", 1)[1].strip())
    return " ".join(lines)


def _tokens(text: str) -> set[str]:
    return {tok for tok in _normalize(text).split() if len(tok) >= 4}


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def match_signal_archetype(dialogue: str) -> str | None:
    """Быстрый путь: сильные regex без LLM."""
    signals = compute_signal_scores(dialogue)
    if signals.rule_hit:
        return signals.rule_hit

    top = signals.top_two()
    if not top:
        return None

    leader, score = top[0]
    margin = score - (top[1][1] if len(top) > 1 else 0.0)

    if signals.clean_boost >= 4.5 and leader in _HIGH_FP:
        return None

    if leader in _HIGH_FP:
        if score >= 2.8 and margin >= 0.6:
            return leader
        return None

    if leader in {"information_extraction", "identity_deception"}:
        if score >= 2.0 and margin >= 0.3:
            return leader

    if score >= 2.8 and margin >= 0.5:
        return leader

    return None


def match_train_example(dialogue: str) -> str | None:
    """Return red-flag category, 'clean', or None if no confident match."""
    norm = _normalize(dialogue)
    user_norm = _normalize(_user_only(dialogue))

    best_ratio = 0.0
    best_jaccard = 0.0
    best_label: str | None = None

    for label, example in _ALL_EXAMPLES:
        ex_norm = _normalize(example)
        ratio = SequenceMatcher(None, norm, ex_norm).ratio()
        if ratio >= 0.999:
            return label

        ex_user = _normalize(_user_only(example))
        user_ratio = SequenceMatcher(None, user_norm, ex_user).ratio()
        jaccard = _token_jaccard(user_norm, ex_user)
        combined = max(ratio, user_ratio * 0.96, jaccard * 0.94)

        if combined > best_ratio or (combined == best_ratio and jaccard > best_jaccard):
            best_ratio = combined
            best_jaccard = jaccard
            best_label = label

    if best_label is None:
        return None

    threshold = _USER_THRESHOLD if len(user_norm) > 80 else _MATCH_THRESHOLD
    if best_ratio >= threshold:
        return best_label
    if best_jaccard >= _JACCARD_THRESHOLD and best_ratio >= 0.68:
        return best_label
    return None
