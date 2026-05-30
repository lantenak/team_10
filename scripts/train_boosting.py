#!/usr/bin/env python3
"""Train LightGBM on train.json with Leave-One-Out CV. Save model + TF-IDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import extract_all_features, extract_user_text

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


def true_label(record: dict) -> str:
    flags = record.get("expected_red_flags", [])
    if not flags:
        return "clean"
    return str(flags[0]["category"])


def format_dialogue(record: dict) -> str:
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in record["messages"])


def main() -> None:
    train_path = ROOT / "app" / "data" / "train.json"
    records: list[dict] = json.loads(train_path.read_text(encoding="utf-8"))

    dialogues = [format_dialogue(r) for r in records]
    labels = [true_label(r) for r in records]

    user_texts = [extract_user_text(d) for d in dialogues]

    tfidf = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 4),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )
    tfidf.fit(user_texts)

    all_feature_dicts = [extract_all_features(d, tfidf_vectorizer=tfidf) for d in dialogues]
    all_keys = sorted({k for fd in all_feature_dicts for k in fd})

    def to_array(fd: dict) -> list[float]:
        return [fd.get(k, 0.0) for k in all_keys]

    X = np.array([to_array(fd) for fd in all_feature_dicts], dtype=np.float32)
    le = LabelEncoder()
    le.fit(CATEGORIES)
    y = le.transform(labels)

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}, Classes: {len(CATEGORIES)}")

    params: dict[str, object] = {
        "objective": "multiclass",
        "num_class": len(CATEGORIES),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 3,
        "min_gain_to_split": 0.01,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    loo = LeaveOneOut()
    correct = 0
    total = 0
    y_pred_loo = np.zeros(len(labels), dtype=int)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=all_keys)
        model = lgb.train(params, dtrain, num_boost_round=200, valid_sets=[dtrain], callbacks=[lgb.log_evaluation(0)])

        pred = model.predict(X_test)
        pred_class = int(np.argmax(pred, axis=1)[0])
        y_pred_loo[test_idx[0]] = pred_class

        if pred_class == y_test[0]:
            correct += 1
        total += 1

    acc = correct / total
    print(f"\nLOO Accuracy: {acc:.3f} ({correct}/{total})")

    for i, (true_c, pred_c) in enumerate(zip(labels, le.inverse_transform(y_pred_loo))):
        if true_c != pred_c:
            print(f"  MISS [{i:02d}] true={true_c:28s} pred={pred_c}")

    dfinal = lgb.Dataset(X, label=y, feature_name=all_keys)
    final_model = lgb.train(
        params, dfinal, num_boost_round=150, valid_sets=[dfinal], callbacks=[lgb.log_evaluation(0)]
    )

    model_path = ROOT / "app" / "model.txt"
    final_model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    vec_path = ROOT / "app" / "tfidf_vectorizer.pkl"
    joblib.dump({"vectorizer": tfidf, "feature_names": all_keys}, vec_path)
    print(f"Vectorizer saved to {vec_path}")

    importances = final_model.feature_importance(importance_type="gain")
    top_idx = np.argsort(importances)[::-1][:20]
    print("\nTop-20 features:")
    for idx in top_idx:
        if importances[idx] > 0:
            print(f"  {all_keys[idx]:50s} {importances[idx]:.2f}")


if __name__ == "__main__":
    main()
