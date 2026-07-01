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


def run(input_dir: str, output_dir: str, workdir: str, workers: int,
        threshold: float) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.dedup import (
        MinhashDedupSignature,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.dedup.minhash import MinhashConfig

    cfg = MinhashConfig()  # VERIFY: set num_buckets/hashes_per_bucket + ngram for ~Jaccard threshold
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
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    run(args.input, args.output, args.workdir, args.workers, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
