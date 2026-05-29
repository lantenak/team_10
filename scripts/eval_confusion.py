#!/usr/bin/env python3
"""Confusion matrix и детальный разбор ошибок на train.json."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models import RED_FLAG_CATEGORIES, load_llm, process_risk_detection  # noqa: E402
from app.signals import compute_signal_scores, format_signal_hints  # noqa: E402

ALL_LABELS = sorted(RED_FLAG_CATEGORIES | {"clean"})


def _true_label(record: dict) -> str:
    flags = record.get("expected_red_flags", [])
    if not flags:
        return "clean"
    return str(flags[0]["category"])


def _dialogue_text(record: dict) -> str:
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in record["messages"])


def _user_snippet(record: dict, *, max_chars: int = 200) -> str:
    parts: list[str] = []
    for msg in record["messages"]:
        if msg.get("role") == "user":
            parts.append(str(msg.get("content", "")))
    text = " | ".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def main() -> None:
    train_path = ROOT / "train.json"
    records: list[dict] = json.loads(train_path.read_text(encoding="utf-8"))

    llm_client = load_llm()
    if not llm_client.api_key:
        print("OPENROUTER_API_KEY не задан.")
        sys.exit(1)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misses: list[dict] = []

    print(f"Confusion analysis: {len(records)} диалогов\n")

    for index, record in enumerate(records, start=1):
        truth = _true_label(record)
        dialogue = _dialogue_text(record)
        signals = compute_signal_scores(dialogue)

        started = time.perf_counter()
        result = process_risk_detection(llm_client, dialogue)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        pred = "clean" if result is None else str(result["category"])
        confusion[truth][pred] += 1

        if truth != pred:
            top = signals.top_two()
            misses.append(
                {
                    "index": index,
                    "session_id": record.get("session_id", "?"),
                    "truth": truth,
                    "pred": pred,
                    "rule_hit": signals.rule_hit,
                    "clean_boost": round(signals.clean_boost, 1),
                    "top_signals": top[:2] if top else [],
                    "hints": format_signal_hints(signals),
                    "snippet": _user_snippet(record),
                    "ms": elapsed_ms,
                },
            )

    print("=== Confusion matrix (rows=true, cols=pred) ===\n")
    header = "true \\ pred".ljust(22) + "".join(label[:12].ljust(14) for label in ALL_LABELS)
    print(header)
    print("-" * len(header))
    for truth in ALL_LABELS:
        row = truth.ljust(22)
        for pred in ALL_LABELS:
            count = confusion[truth][pred]
            row += str(count).ljust(14)
        print(row)

    print(f"\n=== Ошибки ({len(misses)}) ===\n")
    for miss in misses:
        print(f"#{miss['index']:02d} {miss['session_id']}")
        print(f"  true={miss['truth']} pred={miss['pred']}  {miss['ms']}ms")
        print(f"  rule_hit={miss['rule_hit']} clean_boost={miss['clean_boost']}")
        if miss["top_signals"]:
            print(f"  signals={miss['top_signals']}")
        if miss["hints"]:
            print(f"  {miss['hints']}")
        print(f"  user: {miss['snippet']}")
        print()

    by_pair: dict[tuple[str, str], int] = defaultdict(int)
    for miss in misses:
        by_pair[(miss["truth"], miss["pred"])] += 1
    if by_pair:
        print("=== Сводка пар ошибок ===")
        for (truth, pred), count in sorted(by_pair.items(), key=lambda x: -x[1]):
            print(f"  {truth} -> {pred}: {count}")


if __name__ == "__main__":
    main()
