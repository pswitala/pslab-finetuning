#!/usr/bin/env python3
"""Download C4 English subset shard-by-shard for CPT English replay.

C4 (Colossal Clean Crawled Corpus) is CC-BY licensed web text used as English
replay during CPT to prevent catastrophic forgetting of English reasoning.
Target replay fraction: 18% of the CPT mix (configured in build_cpt_mix.py).

C4 has 1024 training shards (~345 MB compressed each, ~350k docs/shard).
10 shards → ~3.5M docs, which exceeds the typical 18% replay target.
No login required — C4 is not gated.

Usage:
    python scripts/ingest/replay_en.py --out-dir data/raw/replay_en
    python scripts/ingest/replay_en.py --out-dir data/raw/replay_en --shards 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.records import today_iso  # noqa: E402

REPO_ID = "allenai/c4"
LANG = "en"
LICENSE = "cc-by"
SNAPSHOT = today_iso()


def list_shards() -> list[str]:
    files = list_repo_files(REPO_ID, repo_type="dataset")
    shards = sorted(
        f for f in files
        if f.startswith(f"{LANG}/c4-train.") and f.endswith(".json.gz")
    )
    return shards


def already_done(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    return set(progress_path.read_text().splitlines())


def mark_done(progress_path: Path, shard: str) -> None:
    with progress_path.open("a") as f:
        f.write(shard + "\n")


def process_shard(shard: str, out_path: Path, shard_index: int) -> int:
    local = hf_hub_download(repo_id=REPO_ID, filename=shard, repo_type="dataset")

    ds = load_dataset("json", data_files=local, split="train")
    n_written = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for row in ds:
            record = {
                "id": f"c4:{LANG}:{shard_index}:{n_written}",
                "source": "c4",
                "url": row.get("url", ""),
                "license": LICENSE,
                "snapshot_date": SNAPSHOT,
                "title": "",
                "text": row.get("text", ""),
                "lang": "en",
                "meta": {
                    "url": row.get("url", ""),
                    "timestamp": str(row.get("timestamp", "")),
                },
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    blob = Path(os.path.realpath(local))
    try:
        Path(local).unlink(missing_ok=True)
        blob.unlink(missing_ok=True)
    except OSError as e:
        print(f"  warning: could not remove {blob}: {e}")

    return n_written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/replay_en")
    ap.add_argument(
        "--shards", type=int, default=None, metavar="N",
        help="Download at most N shards (default: all). 10 shards ≈ 3.5M docs.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / ".progress"

    print(f"Listing shards for {REPO_ID} / {LANG} ...")
    all_shards = list_shards()
    if not all_shards:
        print("No shards found — check dataset name and authentication.")
        return 1

    shards = all_shards[:args.shards] if args.shards is not None else all_shards
    limit_msg = f" (limited to {args.shards})" if args.shards is not None else ""
    print(f"Found {len(all_shards)} shards total{limit_msg}")

    done = already_done(progress_path)
    remaining = [s for s in shards if s not in done]
    if done:
        print(f"Resuming: {len(done)} already done, {len(remaining)} remaining")

    total_written = 0
    for i, shard in enumerate(remaining, 1):
        shard_index = all_shards.index(shard)
        out_path = out_dir / f"part-{shard_index:05d}.jsonl"
        print(f"[{i}/{len(remaining)}] {shard} -> {out_path.name}", end=" ... ", flush=True)
        n = process_shard(shard, out_path, shard_index)
        total_written += n
        mark_done(progress_path, shard)
        print(f"{n} records")

    print(f"Done — {total_written} records across {len(shards)} shards in {out_dir}")
    if args.shards is None and not remaining:
        progress_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
