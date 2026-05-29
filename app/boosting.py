from __future__ import annotations

import pathlib
import typing

import joblib
import lightgbm as lgb
import numpy as np

from app.features import extract_all_features

CATEGORIES = sorted(
    [
        "adversarial_attack",
        "clean",
        "identity_deception",
        "information_extraction",
        "policy_manipulation",
        "scope_violation",
        "transaction_coercion",
    ]
)


class BoostingModel:
    def __init__(self, model_dir: str = "app") -> None:
        base = pathlib.Path(model_dir)

        self._model = lgb.Booster(model_file=str(base / "model.txt"))
        data: dict[str, typing.Any] = joblib.load(base / "tfidf_vectorizer.pkl")
        self._vectorizer = data["vectorizer"]
        self._feature_names: list[str] = data["feature_names"]

    def predict(self, dialogue_text: str) -> dict[str, float]:
        fd = extract_all_features(dialogue_text, tfidf_vectorizer=self._vectorizer)
        row = [fd.get(k, 0.0) for k in self._feature_names]
        probs = self._model.predict(np.array([row], dtype=np.float32))[0]
        return {cat: float(probs[i]) for i, cat in enumerate(CATEGORIES)}

    def top_category(self, dialogue_text: str) -> tuple[str | None, float]:
        probs = self.predict(dialogue_text)
        sorted_cats = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_prob = sorted_cats[0]
        if top_cat == "clean":
            return None, top_prob
        return top_cat, top_prob

    def red_flag_probs(self, dialogue_text: str) -> dict[str, float]:
        probs = self.predict(dialogue_text)
        return {k: v for k, v in probs.items() if k != "clean"}

    def is_suspicious(self, dialogue_text: str, threshold: float = 0.3) -> bool:
        rfp = self.red_flag_probs(dialogue_text)
        return any(v >= threshold for v in rfp.values())


def load_boosting_model(model_dir: str = "app") -> BoostingModel:
    return BoostingModel(model_dir)


def format_boosting_hint(probs: dict[str, float]) -> str:
    sorted_cats = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    red_flag = [(c, p) for c, p in sorted_cats if c != "clean"][:3]
    if not red_flag:
        return ""

    lines: list[str] = []
    top_cat, top_prob = red_flag[0]
    if top_prob >= 0.4:
        lines.append(
            f"ML-модель (LightGBM) с уверенностью {top_prob:.0%} "
            f"считает, что намерение относится к «{top_cat}»."
        )
    elif top_prob >= 0.2:
        parts = ", ".join(f"{c} ({p:.0%})" for c, p in red_flag if p >= 0.1)
        if parts:
            lines.append(f"ML-модель обнаружила подозрительные сигналы: {parts}. Проверь внимательнее.")

    clean_prob = probs.get("clean", 0.0)
    if clean_prob < 0.4:
        lines.append("ML-модель оценивает вероятность clean как низкую — проверь наличие запрещённого намерения.")

    if not lines:
        return ""

    return "АНАЛИЗ ML-МОДЕЛИ (подтверди или отвергни):\n" + "\n".join(lines)
