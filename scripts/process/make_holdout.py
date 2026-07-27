#!/usr/bin/env python3
"""Split catalog records into a train set and a held-out set for closed-book eval.

catalog_eval.py / agentic_eval.py measure recall of facts the model was NEVER trained on.
That requires holding some catalog records OUT of the SFT/CPT data. This script routes each
record to train or holdout by a STABLE hash of its id, so:
  - the split is reproducible (same seed -> same split), and
  - a given record can never leak between the two sets across re-runs.

The holdout dir MUST live outside the `data/catalogs/**` glob that the SFT/CPT builders
read (otherwise held-out records leak back into training). It is written under
`data/holdout/` and an id manifest (`ids.txt`) is emitted so the builders can exclude
those ids by id, independent of file paths (see build_sft_qa.py / build_cpt_mix.py
--exclude-ids).

Usage:
    python scripts/process/make_holdout.py \
        --input "data/catalogs/**/*.jsonl" \
        --train-out data/catalogs_train \
        --holdout-out data/holdout/catalog \
        --fraction 0.02
    # then build SFT/CPT excluding the holdout ids:
    #   build_sft_qa.py --exclude-ids data/holdout/catalog/ids.txt ...
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path


def in_holdout(rec_id: str, fraction: float, seed: int) -> bool:
    """Deterministic per-id routing: True if this id falls in the holdout fraction."""
    h = hashlib.md5(f"{seed}:{rec_id}".encode()).hexdigest()
    # First 8 hex digits -> [0, 1) uniform.
    return (int(h[:8], 16) / 0xFFFFFFFF) < fraction


def iter_jsonl(pattern: str):
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
    ap.add_argument("--input", required=True, help='glob, e.g. "data/catalogs/**/*.jsonl"')
    ap.add_argument("--train-out", required=True, help="dir for the training split")
    ap.add_argument("--holdout-out", required=True, help="dir for the held-out split")
    ap.add_argument("--fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    train_dir, hold_dir = Path(args.train_out), Path(args.holdout_out)
    train_dir.mkdir(parents=True, exist_ok=True)
    hold_dir.mkdir(parents=True, exist_ok=True)
    if hold_dir.resolve() == train_dir.resolve():
        raise SystemExit("--train-out and --holdout-out must be different directories")
    train_fh = (train_dir / "records.jsonl").open("w", encoding="utf-8")
    hold_fh = (hold_dir / "records.jsonl").open("w", encoding="utf-8")
    # Manifest of held-out ids, so downstream builders exclude them by id (path-independent).
    ids_fh = (hold_dir / "ids.txt").open("w", encoding="utf-8")

    n_train = n_hold = n_skip = 0
    try:
        for rec in iter_jsonl(args.input):
            rec_id = rec.get("id")
            if not rec_id:
                n_skip += 1
                continue
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if in_holdout(rec_id, args.fraction, args.seed):
                hold_fh.write(line)
                ids_fh.write(f"{rec_id}\n")
                n_hold += 1
            else:
                train_fh.write(line)
                n_train += 1
    finally:
        train_fh.close()
        hold_fh.close()
        ids_fh.close()

    print(f"train={n_train} holdout={n_hold} skipped(no id)={n_skip}")
    print(f"-> {train_dir}  |  {hold_dir} (+ ids.txt manifest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
