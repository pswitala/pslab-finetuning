#!/usr/bin/env python3
"""Estimate the number of optimizer steps a full training run will take.

Reads a training config, tokenizes its prepared dataset with the model's real
tokenizer, and computes how many packed sequences and optimizer steps result — so you
can size the run (and the eval cadence) before launching a multi-day job.

    steps/epoch = packed_sequences / effective_batch_size
    packed_sequences ≈ total_tokens / max_seq_len       (packing fills full-length seqs)
    effective_batch_size = per_device_batch × grad_accum × num_gpus

This mirrors what the trainer prints as "Total optimization steps", but without loading
the model or starting training. Only meaningful for CPT-style packed runs (packing=true);
for SFT/DPO (packing=false) each example is its own sequence — pass --no-packing.

Usage:
    # Exact count for the full-run settings (override the pilot values in the config):
    python scripts/train/count_steps.py --config configs/cpt.yaml \
        --max-seq-len 4096 --per-device-batch-size 2 --grad-accum 8

    # Fast estimate from a 100k-doc sample (avoids tokenizing the whole corpus):
    python scripts/train/count_steps.py --config configs/cpt.yaml --sample 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_config  # noqa: E402
from cpt import load_text_dataset  # noqa: E402  (reuse the exact loader cpt.py uses)


def count_tokens(ds, tokenizer, num_proc: int, text_field: str) -> int:
    """Total token count over the dataset's text column (no special tokens added).

    Packing concatenates raw document tokens, so we count exactly what packing sees.
    """
    def _lengths(batch):
        enc = tokenizer(batch[text_field], add_special_tokens=False)
        return {"_n": [len(ids) for ids in enc["input_ids"]]}

    counted = ds.map(
        _lengths, batched=True, num_proc=num_proc,
        remove_columns=ds.column_names, desc="Counting tokens",
    )
    return int(sum(counted["_n"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--train-files", default=None,
                    help="glob override (default: train_files from the config)")
    ap.add_argument("--text-field", default="text")
    # Run-shape overrides — default to the config's values so you can model the full run
    # without editing the (currently pilot) config.
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--per-device-batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="default: detected CUDA devices, else 1")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--no-packing", action="store_true",
                    help="treat each example as one sequence (SFT/DPO style)")
    ap.add_argument("--sample", type=int, default=None,
                    help="tokenize only this many random docs and extrapolate (fast)")
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Resolve run shape: CLI override > config > sensible default.
    max_seq_len = args.max_seq_len or cfg.get("max_seq_len", 4096)
    per_device = args.per_device_batch_size or cfg.get("per_device_train_batch_size", 1)
    grad_accum = args.grad_accum or cfg.get("gradient_accumulation_steps", 1)
    epochs = args.epochs if args.epochs is not None else cfg.get("num_train_epochs", 1)
    packing = cfg.get("packing", True) and not args.no_packing

    num_gpus = args.num_gpus
    if num_gpus is None:
        try:
            import torch
            num_gpus = max(1, torch.cuda.device_count())
        except Exception:  # noqa: BLE001
            num_gpus = 1

    train_glob = args.train_files or cfg["train_files"]
    print(f"[count_steps] loading dataset: {train_glob}")
    ds = load_text_dataset(train_glob)
    n_docs = len(ds)
    print(f"[count_steps] documents: {n_docs:,}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)

    if args.sample and args.sample < n_docs:
        sample_ds = ds.shuffle(seed=args.seed).select(range(args.sample))
        sampled_tokens = count_tokens(sample_ds, tokenizer, args.num_proc, args.text_field)
        mean_tokens = sampled_tokens / args.sample
        total_tokens = int(mean_tokens * n_docs)
        print(f"[count_steps] sampled {args.sample:,} docs → mean {mean_tokens:.1f} tok/doc "
              f"(EXTRAPOLATED total)")
    else:
        total_tokens = count_tokens(ds, tokenizer, args.num_proc, args.text_field)
        mean_tokens = total_tokens / n_docs if n_docs else 0.0
        print(f"[count_steps] mean {mean_tokens:.1f} tok/doc (EXACT total)")

    if packing:
        # Packing inserts one separator/EOS token between documents, then slices into
        # full-length sequences and drops the trailing partial one.
        packed_tokens = total_tokens + n_docs
        sequences = packed_tokens // max_seq_len
        seq_desc = f"packed into {max_seq_len}-token sequences"
    else:
        sequences = n_docs
        seq_desc = "one sequence per document (no packing)"

    effective_batch = per_device * grad_accum * num_gpus
    steps_per_epoch = sequences // effective_batch
    total_steps = int(steps_per_epoch * epochs)

    print()
    print("=" * 62)
    print(f"  total tokens          {total_tokens:,}")
    print(f"  {seq_desc}")
    print(f"  sequences             {sequences:,}")
    print(f"  effective batch       {per_device} × {grad_accum} ga × {num_gpus} gpu "
          f"= {effective_batch}")
    print(f"  steps / epoch         {steps_per_epoch:,}")
    print(f"  epochs                {epochs}")
    print(f"  >>> TOTAL STEPS       {total_steps:,}")
    print("=" * 62)

    eval_steps = cfg.get("eval_steps")
    if eval_steps and total_steps:
        n_evals = total_steps // eval_steps
        print(f"  eval every {eval_steps} steps → {n_evals} evals over the run.")
        print(f"  At ~19 min/eval (full val set) that is ~{n_evals * 19 / 60:.1f} h of eval;")
        print(f"  subsample eval_files or raise eval_steps to cut it down.")

    save_steps = cfg.get("save_steps")
    if save_steps and total_steps:
        print(f"  checkpoints every {save_steps} steps → {total_steps // save_steps} saves.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
