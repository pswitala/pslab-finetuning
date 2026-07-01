#!/usr/bin/env python3
"""Stage 2 — Supervised Fine-Tuning (SFT) in Polish.

QLoRA chat fine-tune via Unsloth + TRL SFTTrainer over Polish instruction data +
catalog-grounded synthetic QA (+ a slice of English instructions). Starts from the
merged CPT checkpoint. Loss is computed on responses only.

Input format: jsonl with either
    {"messages": [{"role": "...", "content": "..."}, ...]}    (preferred)
  or {"instruction": ..., "input": ..., "output": ...}        (converted to messages)

Usage:
    python scripts/train/sft.py --config configs/sft.yaml
    python scripts/train/sft.py --config configs/sft.yaml --merge
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_config, load_model_and_tokenizer, load_for_merge, merge_and_save  # noqa: E402


def _to_messages(ex: dict) -> dict:
    if "messages" in ex:
        return ex
    user = ex.get("instruction", "")
    if ex.get("input"):
        user = f"{user}\n\n{ex['input']}"
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": ex.get("output", "")}]}


def load_chat_dataset(files_glob: str, tokenizer, enable_thinking: bool = False):
    from datasets import load_dataset, Dataset
    paths = glob.glob(files_glob, recursive=True)
    if not paths:
        raise FileNotFoundError(f"no files match {files_glob}")

    try:
        ds = load_dataset("json", data_files=paths, split="train")
    except Exception as exc:
        # train.jsonl mixes dolci-sft records {id, messages} with catalog_qa records
        # that also carry {source, license, snapshot_date}. The datasets library infers
        # schema from the first batch and fails to cast later records with extra columns.
        # Fall back to line-by-line loading, keeping only the messages column.
        import json as _json
        print(f"[sft] schema mismatch in source files ({exc.__class__.__name__}: {exc}); "
              "falling back to line-by-line loader (keeps only 'messages' column)")
        rows: list[dict] = []
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if "messages" in rec:
                        rows.append({"messages": rec["messages"]})
        ds = Dataset.from_list(rows)
        print(f"[sft] loaded {len(ds):,} examples via fallback")

    ds = ds.map(_to_messages)

    def render(ex):
        # enable_thinking=False suppresses <think> blocks so they don't appear
        # in training labels. The model was instruct-trained with thinking on by
        # default; turning it off here gives clean Polish instruction-following data.
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older tokenizer versions don't accept enable_thinking; fall back.
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    return ds.map(render, remove_columns=[c for c in ds.column_names if c != "text"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.merge:
        loaded = load_for_merge(cfg)
        merge_and_save(loaded, cfg["output_dir"])
        return 0

    loaded = load_model_and_tokenizer(cfg)
    _maybe_set_chat_template(loaded, cfg)
    print(f"[sft] backend={loaded.backend} base={cfg['base_model']}")

    from trl import SFTTrainer, SFTConfig

    enable_thinking = cfg.get("enable_thinking", False)
    train_ds = load_chat_dataset(cfg["train_files"], loaded.tokenizer, enable_thinking)
    eval_ds = None
    if cfg.get("eval_files"):
        try:
            eval_ds = load_chat_dataset(cfg["eval_files"], loaded.tokenizer, enable_thinking)
        except FileNotFoundError:
            pass

    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        dataset_text_field="text",
        max_length=cfg.get("max_seq_len", 4096),
        packing=cfg.get("packing", False),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        max_steps=cfg.get("max_steps", -1),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 20),
        save_steps=cfg.get("save_steps", 200),
        eval_steps=cfg.get("eval_steps", 200) if eval_ds else None,
        eval_strategy="steps" if eval_ds else "no",
        seed=cfg.get("seed", 3407),
    )
    trainer = SFTTrainer(model=loaded.model, processing_class=loaded.tokenizer,
                         train_dataset=train_ds, eval_dataset=eval_ds, args=sft_cfg)

    # Train on responses only: mask prompt tokens in the loss.
    # Qwen3.6 uses the standard ChatML format (<|im_start|>role\n ... <|im_end|>).
    if cfg.get("train_on_responses_only", True) and loaded.backend == "unsloth":
        try:
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sft] train_on_responses_only (Unsloth) skipped: {exc}")
            # PEFT fallback: TRL DataCollatorForCompletionOnlyLM handles masking.
            try:
                from trl import DataCollatorForCompletionOnlyLM
                response_template = "<|im_start|>assistant\n"
                collator = DataCollatorForCompletionOnlyLM(
                    response_template, tokenizer=loaded.tokenizer)
                trainer.data_collator = collator
            except Exception as exc2:  # noqa: BLE001
                print(f"[sft] DataCollatorForCompletionOnlyLM also skipped: {exc2}")

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[sft] done -> {cfg['output_dir']}")
    return 0


def _maybe_set_chat_template(loaded, cfg) -> None:
    # Qwen3.6-27B ships with its own chat_template.jinja — use it as-is.
    # We no longer override with Unsloth's "qwen3" template since the model's
    # own template handles thinking mode correctly via enable_thinking kwarg.
    pass


if __name__ == "__main__":
    sys.exit(main())
