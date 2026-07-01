#!/usr/bin/env python3
"""Cross-corpus MinHash near-deduplication using datatrove.

Run AFTER pipeline.py, over the union of all cleaned shards, so duplicates that span
sources (e.g. Wikipedia text echoed on the web, legal acts repeated across years) are
removed once. MinHash here is a 4-stage datatrove flow:
  1. compute signatures
  2. find matching buckets
  3. build clusters of duplicates
  4. filter — keep one document per cluster

VERIFY at execution: datatrove MinHash module paths / config class names against the
installed version; they change between releases.

Usage:
    python scripts/process/dedup.py \
        --input data/interim/clean \
        --output data/interim/dedup \
        --workdir data/interim/_minhash --workers 16
"""

from __future__ import annotations

import argparse
import sys


def _num_buckets_for_threshold(threshold: float, hashes_per_bucket: int) -> int:
    """Approximate MinHash-LSH band count for a target Jaccard threshold.

    With `b` bands (num_buckets) of `r` rows (hashes_per_bucket), the LSH match
    probability crosses ~0.5 near t ≈ (1/b)^(1/r), so b ≈ t^(-r). E.g. t=0.8, r=8 -> b=6.
    """
    return max(1, round(threshold ** (-hashes_per_bucket)))


def build_minhash_config(threshold: float, hashes_per_bucket: int, n_grams: int):
    """Build a MinhashConfig tuned to `threshold`, tolerant of datatrove API drift."""
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    num_buckets = _num_buckets_for_threshold(threshold, hashes_per_bucket)
    try:
        cfg = MinhashConfig(n_grams=n_grams, num_buckets=num_buckets,
                            hashes_per_bucket=hashes_per_bucket)
    except TypeError as exc:  # field names differ in this datatrove version
        print(f"[dedup] MinhashConfig kwargs unsupported in this datatrove version "
              f"({exc}); falling back to defaults — verify num_buckets/hashes_per_bucket "
              f"manually to enforce threshold ~{threshold}.")
        cfg = MinhashConfig()
    eff = getattr(cfg, "num_buckets", num_buckets)
    print(f"[dedup] MinHash config: num_buckets={eff} "
          f"hashes_per_bucket={hashes_per_bucket} n_grams={n_grams} "
          f"(~Jaccard threshold {threshold})")
    return cfg


def run(input_dir: str, output_dir: str, workdir: str, workers: int,
        threshold: float, hashes_per_bucket: int = 8, n_grams: int = 5) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.dedup import (
        MinhashDedupSignature,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )

    cfg = build_minhash_config(threshold, hashes_per_bucket, n_grams)
    sig_dir = f"{workdir}/signatures"
    buckets_dir = f"{workdir}/buckets"
    clusters_dir = f"{workdir}/clusters"

    # Stage 1: signatures
    LocalPipelineExecutor(
        pipeline=[JsonlReader(input_dir, glob_pattern="**/*.jsonl.gz"),
                  MinhashDedupSignature(output_folder=sig_dir, config=cfg)],
        tasks=workers, workers=workers,
    ).run()

    # Stage 2: buckets — tasks must be divisible by num_buckets
    LocalPipelineExecutor(
        pipeline=[MinhashDedupBuckets(input_folder=sig_dir,
                                      output_folder=buckets_dir, config=cfg)],
        tasks=cfg.num_buckets, workers=min(workers, cfg.num_buckets),
    ).run()

    # Stage 3: clusters
    LocalPipelineExecutor(
        pipeline=[MinhashDedupCluster(input_folder=buckets_dir,
                                      output_folder=clusters_dir, config=cfg)],
        tasks=1, workers=1,
    ).run()

    # Stage 4: filter -> keep one per cluster
    LocalPipelineExecutor(
        pipeline=[JsonlReader(input_dir, glob_pattern="**/*.jsonl.gz"),
                  MinhashDedupFilter(input_folder=clusters_dir),
                  JsonlWriter(output_folder=output_dir)],
        tasks=workers, workers=workers,
    ).run()
    print(f"dedup complete -> {output_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workdir", default="data/interim/_minhash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="target Jaccard similarity; sets num_buckets via the LSH curve")
    ap.add_argument("--hashes-per-bucket", type=int, default=8)
    ap.add_argument("--n-grams", type=int, default=5)
    args = ap.parse_args()
    run(args.input, args.output, args.workdir, args.workers, args.threshold,
        args.hashes_per_bucket, args.n_grams)
    return 0


if __name__ == "__main__":
    sys.exit(main())
