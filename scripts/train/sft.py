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
from _common import (  # noqa: E402
    load_config, load_model_and_tokenizer, load_for_merge, merge_and_save, wandb_report_to,
)


def _to_messages(ex: dict) -> dict:
    if "messages" in ex:
        return ex   # preserves an optional top-level "tools" list for tool-use samples
    user = ex.get("instruction", "")
    if ex.get("input"):
        user = f"{user}\n\n{ex['input']}"
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": ex.get("output", "")}]}


def _render_chat(tokenizer, messages, tools, enable_thinking: bool) -> str:
    """Render one conversation to text, passing `tools` when present.

    enable_thinking=False suppresses <think> blocks in the labels (instruct models
    generate them by default). Progressively drops kwargs the tokenizer doesn't accept:
    enable_thinking first (older tokenizers), then tools (non-tool-capable templates).
    """
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)   # template predates tool support
            return tokenizer.apply_chat_template(messages, **kwargs)


def load_chat_dataset(files_glob: str, tokenizer, enable_thinking: bool = False):
    """Read chat/tool-use jsonl and render to a flat {'text': ...} dataset.

    Records are read + rendered in Python (not via load_dataset's arrow inference) so the
    deeply-nested, heterogeneous `messages`/`tools`/`tool_calls.arguments` structures —
    which arrow cannot unify across plain-chat and tool-use rows — never reach a columnar
    schema. Only the final flat `text` column is materialized.
    """
    import json as _json
    from datasets import Dataset

    paths = glob.glob(files_glob, recursive=True)
    if not paths:
        raise FileNotFoundError(f"no files match {files_glob}")

    rows: list[dict] = []
    n_tool = 0
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
                ex = _to_messages(rec)
                if "messages" not in ex:
                    continue
                tools = ex.get("tools")
                if tools:
                    n_tool += 1
                rows.append({"text": _render_chat(
                    tokenizer, ex["messages"], tools, enable_thinking)})

    if not rows:
        raise FileNotFoundError(f"no usable records in {files_glob}")
    print(f"[sft] loaded {len(rows):,} examples ({n_tool:,} tool-use) from "
          f"{len(paths)} file(s)")
    return Dataset.from_list(rows)


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
        report_to=wandb_report_to(),
    )
    trainer = SFTTrainer(model=loaded.model, processing_class=loaded.tokenizer,
                         train_dataset=train_ds, eval_dataset=eval_ds, args=sft_cfg)

    # Train on responses only: mask prompt tokens in the loss.
    # Separator tokens are read from config (instruction_part / response_part).
    if cfg.get("train_on_responses_only", True) and loaded.backend == "unsloth":
        instruction_part = cfg.get("instruction_part", "<|im_start|>user\n")
        response_part = cfg.get("response_part", "<|im_start|>assistant\n")
        try:
            from unsloth.chat_templates import train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part=instruction_part,
                response_part=response_part,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sft] train_on_responses_only (Unsloth) skipped: {exc}")
            # PEFT fallback: TRL DataCollatorForCompletionOnlyLM handles masking.
            try:
                from trl import DataCollatorForCompletionOnlyLM
                collator = DataCollatorForCompletionOnlyLM(
                    response_part, tokenizer=loaded.tokenizer)
                trainer.data_collator = collator
            except Exception as exc2:  # noqa: BLE001
                print(f"[sft] DataCollatorForCompletionOnlyLM also skipped: {exc2}")

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[sft] done -> {cfg['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
