#!/usr/bin/env python3
"""Download CulturaX Polish subset shard-by-shard, appending to JSONL and removing
each cached parquet after conversion to free disk space incrementally.

CulturaX uses FLORES-200 language codes — Polish is "pol_Latn", not "pl".
License: ODC-BY (open, commercial-safe).

Requires: pip install huggingface_hub datasets
Authentication: huggingface-cli login (dataset is gated — accept terms on HF first).

Usage:
    python scripts/ingest/culturax_pl.py --out-dir data/raw/culturax_pl
    python scripts/ingest/culturax_pl.py --out-dir data/raw/culturax_pl --shards 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download
from huggingface_hub.errors import GatedRepoError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.records import today_iso  # noqa: E402

REPO_ID = "uonlp/CulturaX"
LANG = "pl"
LICENSE = "odc-by"
SNAPSHOT = today_iso()


def list_shards() -> list[str]:
    files = list_repo_files(REPO_ID, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(f"{LANG}/") and f.endswith(".parquet"))
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

    ds = load_dataset("parquet", data_files=local, split="train")
    n_written = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for row in ds:
            record = {
                "id": f"culturax:{LANG}:{shard_index}:{n_written}",
                "source": "culturax",
                "url": f"https://huggingface.co/datasets/{REPO_ID}",
                "license": LICENSE,
                "snapshot_date": SNAPSHOT,
                "title": "",
                "text": row.get("text", ""),
                "lang": "pl",
                "meta": {"url": row.get("url", ""), "timestamp": row.get("timestamp", "")},
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    # Remove the blob to free disk space; resolve symlink first
    blob = Path(os.path.realpath(local))
    try:
        Path(local).unlink(missing_ok=True)  # snapshot symlink
        blob.unlink(missing_ok=True)          # actual content
    except OSError as e:
        print(f"  warning: could not remove {blob}: {e}")

    return n_written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/culturax_pl")
    ap.add_argument(
        "--shards", type=int, default=None, metavar="N",
        help="Download at most N shards (default: all)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / ".progress"

    print(f"Listing shards for {REPO_ID} / {LANG} ...")
    all_shards = list_shards()
    if not all_shards:
        print("No shards found — check language code and authentication.")
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
        try:
            n = process_shard(shard, out_path, shard_index)
        except GatedRepoError:
            print(
                "\n\nCulturaX is a gated dataset — authentication required:\n"
                "  1. Accept the terms at https://huggingface.co/datasets/uonlp/CulturaX\n"
                "  2. Run:  huggingface-cli login\n"
                "  3. Re-run this script."
            )
            return 1
        total_written += n
        mark_done(progress_path, shard)
        print(f"{n} records")

    print(f"Done — {total_written} records across {len(shards)} shards in {out_dir}")
    if args.shards is None and not remaining:
        # All shards downloaded — clean up progress marker
        progress_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())