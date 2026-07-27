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
import glob
import sys


def _lsh_params_for_threshold(threshold: float,
                              num_permutations: int = 112) -> tuple[int, int]:
    """Pick (num_buckets b, hashes_per_bucket r) approximating a target Jaccard threshold.

    An LSH scheme of `b` bands × `r` rows has match probability 1-(1-s^r)^b for a pair of
    Jaccard similarity s; the S-curve inflects near t ≈ (1/b)^(1/r). We search over r for
    the (b = num_permutations // r, r) whose inflection point is closest to `threshold`,
    keeping the total permutation count b*r near `num_permutations`.

    This replaces the old b ≈ t^(-r) heuristic, which produced far too few permutations
    (t=0.8, r=8 -> only 6 buckets × 8 = 48 perms) for a usable precision/recall curve.
    Standard MinHash uses ~100-256 permutations; the default here (112) matches datatrove.
    """
    best: tuple[float, int, int] | None = None
    for r in range(1, num_permutations + 1):
        b = num_permutations // r
        if b < 1:
            break
        approx_threshold = (1.0 / b) ** (1.0 / r)
        dist = abs(approx_threshold - threshold)
        if best is None or dist < best[0]:
            best = (dist, b, r)
    assert best is not None
    _, num_buckets, hashes_per_bucket = best
    return num_buckets, hashes_per_bucket


def build_minhash_config(threshold: float, num_permutations: int, n_grams: int):
    """Build a MinhashConfig tuned to `threshold` with a proper permutation budget.

    Fails loudly if the installed datatrove's MinhashConfig doesn't accept these kwargs,
    rather than silently falling back to defaults (which would ignore `threshold`).
    """
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    num_buckets, hashes_per_bucket = _lsh_params_for_threshold(threshold, num_permutations)
    try:
        cfg = MinhashConfig(n_grams=n_grams, num_buckets=num_buckets,
                            hashes_per_bucket=hashes_per_bucket)
    except TypeError as exc:  # field names differ in this datatrove version
        raise RuntimeError(
            f"MinhashConfig(n_grams=, num_buckets=, hashes_per_bucket=) rejected by the "
            f"installed datatrove ({exc}). The API changed between releases — update these "
            f"kwargs to match your version instead of silently ignoring --threshold."
        ) from exc
    print(f"[dedup] MinHash config: num_buckets={num_buckets} "
          f"hashes_per_bucket={hashes_per_bucket} "
          f"(total {num_buckets * hashes_per_bucket} permutations) n_grams={n_grams} "
          f"(~Jaccard threshold {threshold})")
    return cfg


# Matches both plain `.jsonl` and gzip `.jsonl.gz` — datatrove's JsonlWriter defaults to
# gzip, but a non-default pipeline may emit plain jsonl; accepting both avoids the
# silent-empty-output bug (reading zero files and "succeeding").
_INPUT_GLOB = "**/*.jsonl*"


def run(input_dir: str, output_dir: str, workdir: str, workers: int,
        threshold: float, num_permutations: int = 112, n_grams: int = 5) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.dedup import (
        MinhashDedupSignature,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )

    n_input = len(glob.glob(f"{input_dir}/{_INPUT_GLOB}", recursive=True))
    if n_input == 0:
        raise FileNotFoundError(
            f"no input shards match {input_dir}/{_INPUT_GLOB} — nothing to dedup. "
            f"Run scripts/process/pipeline.py first, and check its output extension."
        )
    print(f"[dedup] {n_input} input shard(s) under {input_dir}")

    cfg = build_minhash_config(threshold, num_permutations, n_grams)
    sig_dir = f"{workdir}/signatures"
    buckets_dir = f"{workdir}/buckets"
    clusters_dir = f"{workdir}/clusters"

    # Stage 1: signatures
    LocalPipelineExecutor(
        pipeline=[JsonlReader(input_dir, glob_pattern=_INPUT_GLOB),
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
        pipeline=[JsonlReader(input_dir, glob_pattern=_INPUT_GLOB),
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
                    help="target Jaccard similarity; sets the (num_buckets, hashes_per_bucket) "
                         "band split via the LSH S-curve")
    ap.add_argument("--num-permutations", type=int, default=112,
                    help="total MinHash permutations (num_buckets × hashes_per_bucket); "
                         "standard range ~100-256, default 112 (datatrove default)")
    ap.add_argument("--n-grams", type=int, default=5)
    args = ap.parse_args()
    run(args.input, args.output, args.workdir, args.workers, args.threshold,
        args.num_permutations, args.n_grams)
    return 0


if __name__ == "__main__":
    sys.exit(main())
