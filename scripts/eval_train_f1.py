#!/usr/bin/env python3
"""Локальная оценка Macro F1 на train.json через process_risk_detection."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models import (
    RED_FLAG_CATEGORIES,
    load_llm,
    process_risk_detection,
)

ALL_LABELS = sorted(RED_FLAG_CATEGORIES | {"clean"})


def _true_label(record: dict) -> str:
    flags = record.get("expected_red_flags", [])
    if not flags:
        return "clean"
    return str(flags[0]["category"])


def _pred_label(record: dict, llm_client, boosting_model) -> str:  # noqa: ANN001
    dialogue = "\n".join(f"{msg['role']}: {msg['content']}" for msg in record["messages"])
    result = process_risk_detection(llm_client, dialogue, boosting_model=boosting_model)
    if result is None:
        return "clean"
    return str(result["category"])


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores: list[float] = []
    for label in ALL_LABELS:
        tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def main() -> None:
    train_path = ROOT / "train.json"
    records: list[dict] = json.loads(train_path.read_text(encoding="utf-8"))

    llm_client = load_llm()
    if not llm_client.api_key:
        print("OPENROUTER_API_KEY не задан. Скопируйте .env.example → .env и укажите ключ.")
        sys.exit(1)

    try:
        from app.boosting import load_boosting_model

        boosting_model = load_boosting_model()
    except Exception as exc:
        print(f"Boosting unavailable ({exc}), running LLM-only eval.")
        boosting_model = None

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[int] = []

    print(f"Оценка {len(records)} диалогов (model={getattr(llm_client, 'model', '?')})")
    for index, record in enumerate(records, start=1):
        truth = _true_label(record)
        started = time.perf_counter()
        pred = _pred_label(record, llm_client, boosting_model)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        latencies_ms.append(elapsed_ms)
        y_true.append(truth)
        y_pred.append(pred)
        mark = "OK" if truth == pred else "MISS"
        print(f"[{index:02d}/{len(records)}] {mark} true={truth:24s} pred={pred:24s} {elapsed_ms}ms")

    macro = _macro_f1(y_true, y_pred)
    avg_ms = sum(latencies_ms) / len(latencies_ms)
    accuracy = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == pred) / len(records)

    print("\n=== Итог ===")
    print(f"Accuracy:     {accuracy:.3f}")
    print(f"Macro F1:     {macro:.3f}")
    print(f"Avg latency:  {avg_ms:.0f} ms (лимит 5000 ms)")
    print(f"Max latency:  {max(latencies_ms)} ms")

    misses = [(truth, pred) for truth, pred in zip(y_true, y_pred, strict=True) if truth != pred]
    if misses:
        print(f"\nОшибки ({len(misses)}):")
        for truth, pred in misses:
            print(f"  {truth} -> {pred}")

    print("\n=== Per-label ===")
    for label in ALL_LABELS:
        tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred != label)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {label:28s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
