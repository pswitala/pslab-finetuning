#!/usr/bin/env python3
"""Stage 1 — Continued Pretraining (CPT) on Polish + catalog text + EN replay.

QLoRA causal-LM training via Unsloth (TRL SFTTrainer in completion/packing mode over
raw text). Adapts Qwen3.6-27B to Polish while replay data guards against forgetting.

Usage:
    python scripts/train/cpt.py --config configs/cpt.yaml

After training, merge the adapter for the SFT stage:
    python scripts/train/cpt.py --config configs/cpt.yaml --merge
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_config, load_model_and_tokenizer, load_for_merge, merge_and_save  # noqa: E402


def load_text_dataset(files_glob: str):
    """Load parquet/jsonl shards into a HF Dataset with a `text` column."""
    from datasets import load_dataset
    paths = glob.glob(files_glob, recursive=True)
    if not paths:
        raise FileNotFoundError(f"no files match {files_glob} "
                                "(run scripts/process/build_cpt_mix.py first)")
    ext = Path(paths[0]).suffix.lstrip(".")
    fmt = "parquet" if ext == "parquet" else "json"
    return load_dataset(fmt, data_files=paths, split="train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--merge", action="store_true",
                    help="merge trained adapter into 16-bit weights and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.merge:
        loaded = load_for_merge(cfg)
        merge_and_save(loaded, cfg["output_dir"])
        return 0

    loaded = load_model_and_tokenizer(cfg)
    print(f"[cpt] backend={loaded.backend} base={cfg['base_model']}")

    from trl import SFTTrainer, SFTConfig

    train_ds = load_text_dataset(cfg["train_files"])
    eval_ds = None
    if cfg.get("eval_files"):
        try:
            eval_ds = load_text_dataset(cfg["eval_files"])
        except FileNotFoundError:
            print("[cpt] no eval files found; skipping eval")

    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        dataset_text_field="text",
        max_length=cfg.get("max_seq_len", 4096),
        packing=cfg.get("packing", False),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 20),
        save_steps=cfg.get("save_steps", 500),
        eval_steps=cfg.get("eval_steps", 500) if eval_ds else None,
        eval_strategy="steps" if eval_ds else "no",
        seed=cfg.get("seed", 3407),
        report_to=["wandb"] if _wandb_enabled() else [],
    )
    trainer = SFTTrainer(
        model=loaded.model,
        processing_class=loaded.tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[cpt] done -> {cfg['output_dir']} (run with --merge to produce merged/)")
    return 0


def _wandb_enabled() -> bool:
    import os
    return bool(os.environ.get("WANDB_API_KEY"))


if __name__ == "__main__":
    sys.exit(main())
