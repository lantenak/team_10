"""TF-IDF similarity index: nearest train/synthetic neighbor без LightGBM."""

from __future__ import annotations

import functools
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.prompts import FEW_SHOT_EXAMPLES
from app.synthetic_fewshots import EXPANDED_FEW_SHOT_EXAMPLES
from app.train_fewshots import TRAIN_FEW_SHOT_EXAMPLES

_VIOLATION_THRESHOLD = 0.38
_CLEAN_THRESHOLD = 0.48
_VIOLATION_MARGIN = 0.04


@functools.lru_cache(maxsize=1)
def _index() -> tuple[object, np.ndarray, list[str]]:
    examples: list[tuple[str, str]] = (
        TRAIN_FEW_SHOT_EXAMPLES + EXPANDED_FEW_SHOT_EXAMPLES + FEW_SHOT_EXAMPLES
    )
    labels = [label for label, _ in examples]
    texts = [text for _, text in examples]

    base = Path(__file__).resolve().parent
    data: dict = joblib.load(base / "tfidf_vectorizer.pkl")
    vectorizer = data["vectorizer"]
    matrix = vectorizer.transform(texts)
    return vectorizer, matrix, labels


def match_similarity(dialogue: str) -> str | None:
    """Ближайший сосед в TF-IDF пространстве → category или clean."""
    vectorizer, matrix, labels = _index()
    query = vectorizer.transform([dialogue])
    sims = cosine_similarity(query, matrix)[0]

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_label = labels[best_idx]

    if best_label != "clean":
        if best_sim >= _VIOLATION_THRESHOLD:
            return best_label
        return None

    clean_sim = best_sim
    viol_sims = [(float(sims[i]), labels[i]) for i in range(len(labels)) if labels[i] != "clean"]
    if not viol_sims:
        return "clean" if clean_sim >= _CLEAN_THRESHOLD else None

    top_viol_sim, top_viol_label = max(viol_sims, key=lambda x: x[0])
    if top_viol_sim >= _VIOLATION_THRESHOLD and top_viol_sim + _VIOLATION_MARGIN >= clean_sim:
        return top_viol_label

    if clean_sim >= _CLEAN_THRESHOLD and clean_sim > top_viol_sim + _VIOLATION_MARGIN:
        return "clean"

    if top_viol_sim >= _VIOLATION_THRESHOLD:
        return top_viol_label
    return None
