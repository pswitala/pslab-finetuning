#!/usr/bin/env python3
"""Stage 3 — Preference Optimization (DPO) in Polish.

QLoRA DPO via Unsloth + TRL DPOTrainer over Polish preference pairs. Starts from the
merged SFT checkpoint. Set loss_type: orpo in the config to use ORPO instead.

Input format: jsonl with {"prompt": ..., "chosen": ..., "rejected": ...}.
`prompt` may be a plain string or a list of chat messages.

Usage:
    python scripts/train/dpo.py --config configs/dpo.yaml
    python scripts/train/dpo.py --config configs/dpo.yaml --merge
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_config, load_model_and_tokenizer, load_for_merge, merge_and_save  # noqa: E402


def load_pref_dataset(files_glob: str, tokenizer):
    from datasets import load_dataset
    paths = glob.glob(files_glob, recursive=True)
    if not paths:
        raise FileNotFoundError(f"no files match {files_glob}")
    ds = load_dataset("json", data_files=paths, split="train")

    def render(ex):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Conversational format (e.g. Dolci-Instruct-DPO-translated):
        #   prompt = list of {role, content} dicts
        #   chosen / rejected = list of {role, content} dicts (single-item: the response)
        # TRL DPOTrainer >= 0.15 handles this natively — return as-is so the
        # format stays consistent (mixing str prompt with list chosen/rejected breaks TRL).
        if isinstance(prompt, list) and isinstance(chosen, (list, dict)):
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

        # Standard flat format: all are plain strings — also pass through.
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    return ds.map(render)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if cfg.get("load_in_4bit", True):
        # Unsloth must patch the DPO trainer before it is imported/instantiated.
        try:
            from unsloth import PatchDPOTrainer
            PatchDPOTrainer()
        except Exception as exc:  # noqa: BLE001
            print(f"[dpo] PatchDPOTrainer skipped: {exc}")

    if args.merge:
        loaded = load_for_merge(cfg)
        merge_and_save(loaded, cfg["output_dir"])
        return 0

    loaded = load_model_and_tokenizer(cfg)
    print(f"[dpo] backend={loaded.backend} base={cfg['base_model']}")

    from trl import DPOTrainer, DPOConfig

    train_ds = load_pref_dataset(cfg["train_files"], loaded.tokenizer)
    eval_ds = None
    if cfg.get("eval_files"):
        try:
            eval_ds = load_pref_dataset(cfg["eval_files"], loaded.tokenizer)
        except FileNotFoundError:
            pass

    dpo_cfg = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=cfg.get("beta", 0.1),
        loss_type=cfg.get("loss_type", "sigmoid"),
        max_length=cfg.get("max_seq_len", 4096),
        max_prompt_length=cfg.get("max_seq_len", 4096) // 2,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 100) if eval_ds else None,
        eval_strategy="steps" if eval_ds else "no",
        seed=cfg.get("seed", 3407),
    )
    trainer = DPOTrainer(
        model=loaded.model,
        ref_model=None,                      # QLoRA: implicit frozen ref via adapters
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=loaded.tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[dpo] done -> {cfg['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
