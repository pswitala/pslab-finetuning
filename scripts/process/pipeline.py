#!/usr/bin/env python3
"""datatrove processing pipeline: raw text/catalog jsonl -> cleaned, deduped jsonl.

Stages:
  1. Read jsonl (corpus dumps or catalog ingest output).
  2. fastText language-ID -> keep pl with confidence >= threshold.
  3. Gopher-style + Polish-tuned quality filter (scripts/process/quality_pl.py).
  4. MinHash near-dedup (document level).
  5. Write cleaned jsonl shards.

Dedup should run ACROSS all corpora together so Wikipedia/legal text that appears in
multiple sources is not double-counted.

VERIFY at execution: datatrove API names (imports below) against the installed
version — datatrove's module paths shift between releases. Adjust imports if needed.

Usage (local executor, single machine):
    python scripts/process/pipeline.py \
        --input "data/raw/**/*.jsonl" \
        --output data/interim/clean \
        --workers 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process.quality_pl import assess  # noqa: E402


def build_pipeline(input_glob: str, output_dir: str, lang_threshold: float):
    """Construct the datatrove pipeline steps. Imports are local so the module can be
    imported (e.g. for tests) even when datatrove isn't installed."""
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter as DTJsonlWriter
    from datatrove.pipeline.filters import (
        LanguageFilter,
        GopherQualityFilter,
        GopherRepetitionFilter,
    )
    from datatrove.pipeline.base import PipelineStep
    from datatrove.data import Document

    class PolishQualityFilter(PipelineStep):
        """Wrap scripts/process/quality_pl.assess as a datatrove filter step."""
        name = "🇵🇱 Polish quality"

        def run(self, data, rank: int = 0, world_size: int = 1):
            for doc in data:
                res = assess(doc.text)
                if res.keep:
                    yield doc
                else:
                    self.stat_update(f"dropped:{res.reason.split('(')[0]}")

    parts = Path(input_glob).parts
    star_idx = next(i for i, p in enumerate(parts) if "*" in p)
    data_folder = str(Path(*parts[:star_idx]))
    glob_pattern = str(Path(*parts[star_idx:]))

    return [
        JsonlReader(data_folder=data_folder, glob_pattern=glob_pattern),
        LanguageFilter(languages=["pl"], language_threshold=lang_threshold),
        GopherRepetitionFilter(),
        GopherQualityFilter(),
        PolishQualityFilter(),
        DTJsonlWriter(output_folder=output_dir),
    ]


def run(input_glob: str, output_dir: str, workers: int, lang_threshold: float) -> None:
    from datatrove.executor import LocalPipelineExecutor

    steps = build_pipeline(input_glob, output_dir, lang_threshold)
    executor = LocalPipelineExecutor(pipeline=steps, tasks=workers, workers=workers,
                                     logging_dir=f"{output_dir}/_logs")
    executor.run()
    print("\nNOTE: run scripts/process/dedup.py next for cross-corpus MinHash dedup.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help='glob, e.g. "data/raw/**/*.jsonl"')
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lang-threshold", type=float, default=0.7)
    args = ap.parse_args()
    run(args.input, args.output, args.workers, args.lang_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
