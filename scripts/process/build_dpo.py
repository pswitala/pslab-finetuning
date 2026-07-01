#!/usr/bin/env python3
"""Build + validate the DPO preference dataset.

Previously DPO data was copied from the Dolci-DPO dump verbatim (README `cp ...`) with no
validation — but that dump contains malformed pairs (empty/degenerate responses, chosen ==
rejected). This step filters those out and writes a clean train/val split that
scripts/train/dpo.py can consume directly.

Accepted input per line: {"prompt", "chosen", "rejected"} where each field is a plain
string OR a list of {role, content} chat messages (both are passed through unchanged;
validation looks at the extracted text).

Usage:
    python scripts/process/build_dpo.py \
        --input data/raw/dolci-dpo-pl/data.jsonl \
        --out data/processed/dpo --val 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def as_text(x) -> str:
    """Extract comparable text from a string or a list of chat messages."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return " ".join(
            str(m.get("content", "")) for m in x if isinstance(m, dict)).strip()
    if isinstance(x, dict):
        return str(x.get("content", "")).strip()
    return ""


def valid_pair(rec: dict, min_chars: int) -> bool:
    """Structural quality gate for a preference pair."""
    if not all(k in rec for k in ("prompt", "chosen", "rejected")):
        return False
    p, c, r = as_text(rec["prompt"]), as_text(rec["chosen"]), as_text(rec["rejected"])
    if not p or not c or not r:
        return False
    if c == r:                       # no preference signal
        return False
    if len(c) < min_chars:           # degenerate chosen response
        return False
    return True


def iter_jsonl(pattern: str):
    import glob
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of raw DPO jsonl")
    ap.add_argument("--out", required=True, help="output dir (writes train.jsonl, val.jsonl)")
    ap.add_argument("--val", type=int, default=500, help="held-out pairs for eval")
    ap.add_argument("--min-chars", type=int, default=2,
                    help="drop pairs whose chosen response is shorter than this")
    args = ap.parse_args()

    kept, dropped = [], 0
    seen: set[str] = set()
    for rec in iter_jsonl(args.input):
        if not valid_pair(rec, args.min_chars):
            dropped += 1
            continue
        key = as_text(rec["prompt"]) + "\x00" + as_text(rec["chosen"])
        if key in seen:              # exact-duplicate pair
            dropped += 1
            continue
        seen.add(key)
        kept.append({"prompt": rec["prompt"], "chosen": rec["chosen"],
                     "rejected": rec["rejected"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_val = min(args.val, len(kept) // 10)   # never let val exceed 10% of data
    val, train = kept[:n_val], kept[n_val:]
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"kept={len(kept)} dropped={dropped} -> train={len(train)} val={len(val)}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
