#!/usr/bin/env python3
"""Phase 6 — evaluation via EleutherAI lm-evaluation-harness.

Runs three suites and writes results under eval/results/:
  - polish       : Full Polish task suite
  - polish_quick : Fast 2-task subset for intermediate checks
  - english      : English-retention check (MMLU/HellaSwag/ARC) — compare to base model
  - (catalog knowledge is a custom closed-book set — see scripts/eval/catalog_eval.py)

PREFER --peft over --model <merged-dir> when the merged model lives on NTFS (/mnt/c/).
Loading from NTFS via WSL is ~80x slower than the HF cache on the Linux filesystem.
With --peft the base model loads from HF cache and the small adapter is applied on top.

Usage:
    # Fast: base from HF cache + adapter (no merge needed)
    python scripts/eval/run_eval.py --peft models/cpt --suite polish_quick

    # After full pipeline, evaluate merged model (save it to Linux fs first):
    python scripts/eval/run_eval.py --model models/dpo/merged --suite polish
    python scripts/eval/run_eval.py --model Qwen/Qwen3.6-27B --suite english   # baseline

    # vllm backend (higher throughput, merged model only — incompatible with --peft):
    python scripts/eval/run_eval.py --model models/dpo/merged --suite polish --backend vllm
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# VERIFY these task ids against `lm-eval --tasks list`.
TASKS = {
    "polish": [
        "belebele_pol_Latn",          # reading comprehension, 900 questions
        "arc_challenge_mt_pl",         # science reasoning (machine-translated)
        "global_mmlu_full_pl",         # broad knowledge, 57 subjects — use --limit for speed
        "global_piqa_completions_pol_latn",  # physical commonsense
    ],
    "polish_quick": [
        "belebele_pol_Latn",           # ~900 questions, fast
        "arc_challenge_mt_pl",          # ~1172 questions
    ],
    "english": [
        "mmlu",
        "hellaswag",
        "arc_challenge",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", help="HF model id or path to merged dir")
    grp.add_argument("--peft", metavar="ADAPTER_DIR",
                     help="Load base model from HF cache + apply this PEFT adapter. "
                          "Faster than --model when the merged dir is on NTFS.")
    ap.add_argument("--base-model", default=None,
                    help="Base model id when using --peft (default: read from adapter config)")
    ap.add_argument("--suite", choices=list(TASKS), required=True)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="0 = full; >0 = quick subset")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="lm-eval inference backend: hf (default) or vllm")
    args = ap.parse_args()

    tasks = ",".join(TASKS[args.suite])
    out_dir = Path(args.out) / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "vllm" and args.peft:
        ap.error("--backend vllm is incompatible with --peft; use a merged model with --model")

    if args.peft:
        # Derive base model from adapter_config.json if not explicitly provided
        base = args.base_model
        if base is None:
            import json
            adapter_cfg = Path(args.peft) / "adapter_config.json"
            if adapter_cfg.exists():
                base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
            else:
                ap.error("--base-model required when adapter_config.json is missing")
        model_args = f"pretrained={base},dtype=bfloat16,peft={args.peft}"
    else:
        model_args = f"pretrained={args.model},dtype=bfloat16"

    if args.backend == "vllm":
        # tensor_parallel_size=1 prevents vllm from attempting multi-GPU sharding
        vllm_args = f"pretrained={args.model},dtype=bfloat16,tensor_parallel_size=1"
        cmd = [
            "lm-eval", "--model", "vllm",
            "--model_args", vllm_args,
            "--tasks", tasks,
            "--batch_size", str(args.batch_size),
            "--output_path", str(out_dir),
        ]
    else:
        cmd = [
            "lm-eval", "--model", "hf",
            "--model_args", model_args,
            "--tasks", tasks,
            "--batch_size", str(args.batch_size),
            "--output_path", str(out_dir),
        ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    print("[eval]", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
