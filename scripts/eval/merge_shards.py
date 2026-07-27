#!/usr/bin/env python3
"""Combine the per-shard outputs of a data-parallel eval fan-out into one result.

catalog_eval.py / agentic_eval.py with --num-shards N write <out>/shard0..shardN-1, each
holding summary.json + details.jsonl. This concatenates the details and recomputes the
summary as an n-weighted average of the shard summaries, so the numbers match what a single
un-sharded run would have produced.

Usage:
    python scripts/eval/merge_shards.py --out eval/results/catalog
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge(out_dir: Path) -> dict:
    shards = sorted(out_dir.glob("shard*"), key=lambda p: int(p.name[5:]))
    if not shards:
        raise SystemExit(f"no shard*/ directories under {out_dir} — was --num-shards used?")

    summaries, details = [], []
    for shard in shards:
        summary_path = shard / "summary.json"
        if not summary_path.exists():
            raise SystemExit(f"{summary_path} missing — shard {shard.name} did not finish")
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        detail_path = shard / "details.jsonl"
        if detail_path.exists():
            details.extend(line for line in
                           detail_path.read_text(encoding="utf-8").splitlines() if line.strip())

    total = sum(s.get("n", 0) for s in summaries)
    if not total:
        raise SystemExit("all shards are empty")

    # Numeric fields are per-item rates, so weight each shard by its item count. Non-numeric
    # fields (model, backend, scorer, ...) are identical across shards; take the first.
    merged: dict = {}
    for key, value in summaries[0].items():
        if key == "n":
            merged["n"] = total
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            merged[key] = value
        else:
            merged[key] = sum(s.get(key, 0) * s.get("n", 0) for s in summaries) / total
    merged["shards"] = len(shards)

    (out_dir / "summary.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "details.jsonl").write_text("\n".join(details), encoding="utf-8")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="the --out directory the sharded run used, e.g. eval/results/catalog")
    args = ap.parse_args()

    out_dir = Path(args.out)
    merged = merge(out_dir)
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
