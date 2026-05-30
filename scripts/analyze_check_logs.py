#!/usr/bin/env python3
"""Сводка по logs/check_requests.jsonl после прогона eval на сервере."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "logs" / "check_requests.jsonl"


def main() -> None:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    if not log_path.is_file():
        print(f"Нет файла {log_path}. Включите DEBUG_LOG_CHECKS=true и дождитесь eval.")
        sys.exit(1)

    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    print(f"Записей: {len(rows)}")
    preds = Counter(r.get("predicted_category") or "clean" for r in rows)
    print("\nPredicted distribution:")
    for cat, count in preds.most_common():
        print(f"  {cat:28s} {count}")

    verbatim = sum(1 for r in rows if r.get("verbatim_train_hit"))
    print(f"\nVerbatim train hits: {verbatim}/{len(rows)}")

    paths = Counter()
    for r in rows:
        if r.get("verbatim_train_hit"):
            paths["verbatim"] += 1
        elif r.get("signals_rule_hit"):
            paths["rule_hit"] += 1
        elif r.get("predicted_category"):
            paths["llm/heuristic"] += 1
        else:
            paths["clean/null"] += 1
    print("\nDecision path (approx):")
    for name, count in paths.most_common():
        print(f"  {name:20s} {count}")

    avg_ms = sum(r.get("processing_time_ms", 0) for r in rows) / len(rows)
    over5 = sum(1 for r in rows if r.get("processing_time_ms", 0) > 5000)
    print(f"\nAvg latency: {avg_ms:.0f} ms")
    print(f"Over 5000 ms: {over5} ({100 * over5 / len(rows):.1f}%)")

    print("\nУникальных session_ref:", len({r.get("session_ref") for r in rows}))


if __name__ == "__main__":
    main()
