#!/usr/bin/env python3
"""Assemble the final CPT training mixture.

Combines cleaned Polish text (general corpora + catalog text) with an English/code
replay stream at a target ratio (default ~18% EN) to mitigate catastrophic forgetting,
and writes sharded parquet with a `text` field + `domain`/`license`/`source` tags.

Tokenization + sequence packing is handled by the trainer (Unsloth `packing=True`),
so this step only curates and mixes documents — it does NOT tokenize.

Inputs are jsonl with at least {"text": ...}; catalog/ingest records also carry
`license`, `source`, `snapshot_date` which are preserved.

Usage:
    python scripts/process/build_cpt_mix.py \
        --pl "data/interim/dedup/**/*.jsonl" \
        --en "data/raw/replay_en/**/*.jsonl" \
        --out data/processed/cpt \
        --replay-fraction 0.18 --val-fraction 0.005 \
        --commercial-safe
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

COMMERCIAL_SAFE_PREFIXES = ("cc0", "cc-by", "public-domain", "pddl", "odc-by",
                            "apache", "mit")


def iter_jsonl(globs: list[str]):
    for g in globs:
        for path in glob.glob(g, recursive=True):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue


def license_ok(rec: dict, commercial_safe: bool) -> bool:
    if not commercial_safe:
        return True
    lic = str(rec.get("license", "unknown")).lower()
    return lic != "unknown" and lic.startswith(COMMERCIAL_SAFE_PREFIXES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pl", nargs="+", required=True, help="glob(s) of Polish jsonl")
    ap.add_argument("--en", nargs="*", default=[], help="glob(s) of English replay jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-fraction", type=float, default=0.18)
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--shard-size", type=int, default=100_000, help="docs per shard")
    ap.add_argument("--commercial-safe", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        print(f"pyarrow required: {exc}")
        return 1

    rng = random.Random(args.seed)
    out_train = Path(args.out) / "train"
    out_val = Path(args.out) / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # First pass: count Polish docs to size the EN replay stream.
    pl_docs = []
    skipped_lic = 0
    for rec in iter_jsonl(args.pl):
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        if args.commercial_safe and not license_ok(rec, True):
            skipped_lic += 1
            continue
        pl_docs.append({"text": text, "domain": "pl",
                        "source": rec.get("source", "corpus"),
                        "license": rec.get("license", "unknown")})
    print(f"Polish docs kept: {len(pl_docs)} (license-skipped: {skipped_lic})")

    # Size EN replay to hit the target fraction: en/(pl+en) = replay_fraction.
    rf = args.replay_fraction
    target_en = int(len(pl_docs) * rf / (1 - rf)) if rf < 1 else 0
    en_docs = []
    if target_en and args.en:
        for rec in iter_jsonl(args.en):
            text = (rec.get("text") or "").strip()
            if text:
                en_docs.append({"text": text, "domain": "en_replay",
                                "source": rec.get("source", "replay"),
                                "license": rec.get("license", "unknown")})
            if len(en_docs) >= target_en:
                break
        print(f"English replay docs: {len(en_docs)} (target {target_en})")
    elif target_en:
        print(f"WARNING: wanted {target_en} EN replay docs but --en not provided; "
              f"proceeding Polish-only (higher forgetting risk).")

    docs = pl_docs + en_docs
    rng.shuffle(docs)

    n_val = int(len(docs) * args.val_fraction)
    val, train = docs[:n_val], docs[n_val:]
    print(f"train={len(train)} val={len(val)}")

    def write_shards(rows: list[dict], out_dir: Path) -> None:
        for i in range(0, len(rows), args.shard_size):
            shard = rows[i:i + args.shard_size]
            table = pa.Table.from_pylist(shard)
            pq.write_table(table, out_dir / f"part-{i // args.shard_size:05d}.parquet")

    write_shards(train, out_train)
    write_shards(val, out_val)
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
