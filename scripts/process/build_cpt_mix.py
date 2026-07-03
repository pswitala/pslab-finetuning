#!/usr/bin/env python3
"""Assemble the final CPT training mixture.

Combines cleaned Polish text (general corpora + catalog text) with an English/code
replay stream at a target ratio (default ~18% EN) to mitigate catastrophic forgetting,
and writes sharded parquet with a `text` field + `domain`/`license`/`source` tags.

Tokenization + sequence packing is handled by the trainer (Unsloth `packing=True`),
so this step only curates and mixes documents — it does NOT tokenize.

Inputs are jsonl with at least {"text": ...}; catalog/ingest records also carry
`license`, `source`, `snapshot_date` which are preserved.

Memory model: STREAMING. Documents are never all held in RAM. We do a counting pass to
size the EN replay, then a single writing pass that assigns each doc to a random shard
(approximate global shuffle) and Bernoulli-samples EN down to the target fraction. This
scales to the ~400 GB+ corpus, unlike an in-memory shuffle.

Usage:
    python scripts/process/build_cpt_mix.py \
        --pl "data/interim/dedup/**/*.jsonl*" \
        --en "data/raw/replay_en/**/*.jsonl" \
        --out data/processed/cpt \
        --replay-fraction 0.18 --val-fraction 0.005 \
        --commercial-safe

    NOTE: use *.jsonl* (not *.jsonl) for datatrove outputs — they are gzipped (.jsonl.gz).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.records import is_commercial_safe  # noqa: E402

_SCHEMA_FIELDS = ("text", "domain", "source", "license")


def _open_text(path: str):
    """Open plain OR gzipped jsonl transparently.

    datatrove (pipeline.py / dedup.py) writes gzipped shards (`*.jsonl.gz`); catalog/ingest
    outputs are plain `.jsonl`. A plain open() on a .gz silently yields no parseable lines,
    which is exactly how the entire deduped web corpus got dropped from the CPT mix.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_jsonl(globs: list[str]):
    for g in globs:
        for path in glob.glob(g, recursive=True):
            with _open_text(path) as fh:
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
    return is_commercial_safe(rec.get("license", "unknown"))


def count_docs(globs: list[str], commercial_safe: bool) -> tuple[int, int]:
    """Stream-count docs with non-empty text (and, optionally, a safe license).

    Returns (kept, license_skipped).
    """
    kept = skipped = 0
    for rec in iter_jsonl(globs):
        if not (rec.get("text") or "").strip():
            continue
        if commercial_safe and not license_ok(rec, True):
            skipped += 1
            continue
        kept += 1
    return kept, skipped


class ShardedParquetWriter:
    """Write rows to N parquet shards with bounded memory (buffered row groups)."""

    def __init__(self, out_dir: Path, num_shards: int, buffer_rows: int = 10_000):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self._pa = pa
        self._pq = pq
        self._schema = pa.schema([(f, pa.string()) for f in _SCHEMA_FIELDS])
        out_dir.mkdir(parents=True, exist_ok=True)
        self.num_shards = max(1, num_shards)
        self.buffer_rows = buffer_rows
        self._writers: list = [None] * self.num_shards
        self._buffers: list[list[dict]] = [[] for _ in range(self.num_shards)]
        self._paths = [out_dir / f"part-{i:05d}.parquet" for i in range(self.num_shards)]
        self.total = 0

    def write(self, shard: int, row: dict) -> None:
        buf = self._buffers[shard]
        buf.append({f: row.get(f, "") for f in _SCHEMA_FIELDS})
        self.total += 1
        if len(buf) >= self.buffer_rows:
            self._flush(shard)

    def _flush(self, shard: int) -> None:
        buf = self._buffers[shard]
        if not buf:
            return
        if self._writers[shard] is None:
            self._writers[shard] = self._pq.ParquetWriter(
                str(self._paths[shard]), self._schema)
        table = self._pa.Table.from_pylist(buf, schema=self._schema)
        self._writers[shard].write_table(table)
        buf.clear()

    def close(self) -> None:
        for i in range(self.num_shards):
            self._flush(i)
            if self._writers[i] is not None:
                self._writers[i].close()
        # Drop empty shard files that never received rows.
        for i in range(self.num_shards):
            if self._writers[i] is None and self._paths[i].exists():
                self._paths[i].unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pl", nargs="+", required=True, help="glob(s) of Polish jsonl")
    ap.add_argument("--en", nargs="*", default=[], help="glob(s) of English replay jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-fraction", type=float, default=0.18)
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--shard-size", type=int, default=100_000, help="target docs per shard")
    ap.add_argument("--commercial-safe", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"pyarrow required: {exc}")
        return 1

    rng = random.Random(args.seed)

    # Pass 1 — count (streaming) to size the EN replay stream.
    n_pl, skipped_lic = count_docs(args.pl, args.commercial_safe)
    print(f"Polish docs kept: {n_pl} (license-skipped: {skipped_lic})")
    if n_pl == 0:
        print("no Polish docs — nothing to do")
        return 1

    rf = args.replay_fraction
    target_en = int(n_pl * rf / (1 - rf)) if 0 < rf < 1 else 0
    p_en = 0.0
    n_en_total = 0
    if target_en and args.en:
        n_en_total, _ = count_docs(args.en, False)  # replay isn't license-filtered
        p_en = min(1.0, target_en / n_en_total) if n_en_total else 0.0
        print(f"English replay: target {target_en} of {n_en_total} available "
              f"(accept prob {p_en:.4f})")
    elif target_en:
        print(f"WARNING: wanted {target_en} EN replay docs but --en not provided; "
              f"proceeding Polish-only (higher forgetting risk).")

    # Size the shard pool from the estimated total, then split train/val by draw.
    est_total = n_pl + min(target_en, n_en_total)
    num_train_shards = max(1, math.ceil(
        est_total * (1 - args.val_fraction) / args.shard_size))
    num_val_shards = max(1, math.ceil(
        est_total * args.val_fraction / args.shard_size))

    out = Path(args.out)
    train_w = ShardedParquetWriter(out / "train", num_train_shards)
    val_w = ShardedParquetWriter(out / "val", num_val_shards)

    def emit(row: dict) -> None:
        if rng.random() < args.val_fraction:
            val_w.write(rng.randrange(val_w.num_shards), row)
        else:
            train_w.write(rng.randrange(train_w.num_shards), row)

    # Pass 2 — write (streaming). Random shard assignment approximates a global shuffle.
    for rec in iter_jsonl(args.pl):
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        if args.commercial_safe and not license_ok(rec, True):
            continue
        emit({"text": text, "domain": "pl",
              "source": rec.get("source", "corpus"),
              "license": rec.get("license", "unknown")})

    if p_en > 0:
        for rec in iter_jsonl(args.en):
            text = (rec.get("text") or "").strip()
            if not text or rng.random() >= p_en:
                continue
            emit({"text": text, "domain": "en_replay",
                  "source": rec.get("source", "replay"),
                  "license": rec.get("license", "unknown")})

    train_w.close()
    val_w.close()
    print(f"train={train_w.total} val={val_w.total}")
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
