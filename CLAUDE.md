# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end QLoRA fine-tuning pipeline for Qwen3.6-27B targeting Polish language fluency and knowledge injection from Polish open-data catalogs. Knowledge is baked into model weights (no RAG). Final artifacts are GGUF-quantized models for llama.cpp/Ollama.

Hardware target: 4× NVIDIA RTX 6000 Pro Blackwell, 96 GB VRAM each, CUDA 12.8, compute capability sm_120. No NVLink — the cards are PCIe-connected, which is why training scales via DDP (one full replica per card, only LoRA gradients all-reduced) rather than FSDP/DeepSpeed, and why eval fans out data-parallel rather than using tensor parallelism.

## Commands

### Verify environment before any training
```bash
python scripts/check_env.py
# Expect: torch 2.10.0, CUDA 12.8, sm_120, ~96 GB VRAM, bf16 matmul OK
# requirements.txt pins nvidia-*-cu12==12.8.x — install torch from the cu128 index, not cu129.
```

### Install dependencies
```bash
pip install -r requirements.txt
# Note: torch, transformers, vllm, flashinfer already in vllm venv
# Never install flash-attn — incompatible with Blackwell/flashinfer
```

### Data ingestion (no GPU required)
```bash
python scripts/ingest/sejm_isap.py    # Polish legal acts → data/catalogs/isap/
python scripts/ingest/dane_gov.py     # Open-data descriptions → data/catalogs/dane_gov/
python scripts/ingest/gus_bdl.py     # GUS statistics → data/catalogs/gus_bdl/
```

### Data processing (CPU-only)
```bash
python scripts/process/pipeline.py   # lang-ID + quality filter → data/interim/clean/
python scripts/process/dedup.py      # MinHash dedup → data/interim/dedup/
python scripts/process/build_cpt_mix.py  # → data/processed/cpt/
python scripts/process/build_sft_qa.py  # → data/processed/sft/
```

### Training (each stage produces adapters, then merged weights)

Multi-GPU: `make cpt NPROC=4` / `make sft NPROC=4` / `make dpo NPROC=4` run 4-way DDP via
`torchrun`. `--merge` must stay single-process (it raises SystemExit under a launcher).
The `distributed:` block in each config controls the effective-batch policy — `constant`
(default) divides `gradient_accumulation_steps` by the world size so multi-GPU runs reproduce
the single-GPU optimizer math.

```bash
python scripts/train/cpt.py --config configs/cpt.yaml          # Stage 1: CPT
python scripts/train/cpt.py --config configs/cpt.yaml --merge  # Merge → models/cpt/merged/

python scripts/train/sft.py --config configs/sft.yaml          # Stage 2: SFT
python scripts/train/sft.py --config configs/sft.yaml --merge  # Merge → models/sft/merged/

python scripts/train/dpo.py --config configs/dpo.yaml          # Stage 3: DPO
python scripts/train/dpo.py --config configs/dpo.yaml --merge  # Merge → models/dpo/merged/
```

### Pilot run (strongly recommended before full training)
```bash
# Add max_steps=200 to any training config to do a quick smoke test
```

### Evaluation
```bash
python scripts/eval/run_eval.py --model models/dpo/merged --suite polish
python scripts/eval/run_eval.py --model models/dpo/merged --suite english  # retention check
python scripts/eval/catalog_eval.py   # closed-book catalog knowledge
```

### Export to GGUF
```bash
python scripts/train/export_gguf.py --config configs/dpo.yaml --quants Q4_K_M Q5_K_M Q8_0
python scripts/eval/smoke_gguf.py --gguf models/gguf/model-Q4_K_M.gguf
```

## Architecture

### Model Architecture: Qwen3.6-27B (critical)

Qwen3.6-27B uses a **hybrid SSM architecture**, not pure transformer attention:
- 64 layers = 16 × (3× Gated DeltaNet SSM + 1× Gated Attention) repeating blocks
- **75% of layers are linear-attention/SSM** (`linear_attn.*`)
- **25% are standard attention** (`self_attn.*`)

**LoRA must target both layer types.** The configs/cpt.yaml `target_modules` includes both `self_attn.*` and `linear_attn.*` projection names. Ignoring SSM layers severely limits adaptation capacity.

The model is also a **Vision-Language Model** (`Qwen3_5ForConditionalGeneration`). The vision encoder is automatically frozen in `scripts/train/_common.py` since this is text-only fine-tuning.

**Thinking mode** is disabled via `enable_thinking=False` in SFT and DPO — the base instruct model generates `<think>` blocks by default, which must be suppressed for instruction-following tasks.

**No base checkpoint exists** — only the instruct checkpoint is available. This means CPT uses a conservative learning rate (3e-5) and 18% English replay to avoid destroying alignment.

### Training Pipeline

```
Polish APIs (Sejm/ISAP, dane.gov.pl, GUS BDL)
    → data/catalogs/  (records carry source/license/snapshot_date; GUS BDL alone is ~5.5M)
    → make_holdout.py splits this into data/catalogs_train/ + data/catalogs_holdout/
      ALL dataset builders must read data/catalogs_train/, never data/catalogs/ (holdout leak)

HuggingFace corpora (HPLT 2.0 PL, Wikipedia PL, CulturaX PL, Dolci SFT/DPO, EN replay)
    → data/raw/  (~500+ GB, gitignored)

pipeline.py (fastText lang-ID → Gopher quality filter → Polish heuristics)
    → data/interim/clean/

dedup.py (4-stage MinHash cross-corpus dedup, Jaccard threshold 0.8)
    → data/interim/dedup/

build_cpt_mix.py / build_sft_qa.py
    → data/processed/{cpt,sft,dpo}/

cpt.py → models/cpt/merged/
sft.py → models/sft/merged/
dpo.py → models/dpo/merged/

export_gguf.py → models/gguf/*.gguf (Q4_K_M, Q5_K_M, Q8_0, f16)
```

### Shared Infrastructure (`scripts/train/_common.py`)

All training scripts import shared utilities:
- `load_config(path)` — reads YAML configs
- `LoadedModel` wrapper — tries Unsloth first, falls back to PEFT automatically
- `_resolve_target_modules()` — auto-detects LoRA targets across SSM and attention layers
- `_maybe_freeze_vision_encoder()` — freezes vision parameters; called twice on the PEFT path
  (before and after `get_peft_model`), because LoRA otherwise lands in the vision tower and
  those adapters become DDP "unused parameters"
- `dist_env()` / `is_main_process()` / `rank0_print()` — launcher-env helpers
- `_resolve_device_map()` — `{"": local_rank}` under a launcher, `auto` single-process
- `distributed_training_args()` — world-size-aware trainer kwargs (effective-batch policy,
  DDP knobs); splatted into every stage's `SFTConfig`/`DPOConfig`

### Config System

Three YAML configs control all hyperparameters:
- `configs/cpt.yaml` — CPT: r=64, α=128, lr=3e-5, 1 epoch, packing=True
- `configs/sft.yaml` — SFT: r=32, α=64, lr=2e-4, 3 epochs, loss on responses only
- `configs/dpo.yaml` — DPO: r=32, α=64, lr=5e-6, 1 epoch, prompt/chosen/rejected format

### Catalog Record Schema (`scripts/common/records.py`)

All ingested records carry: `text`, `source`, `license`, `snapshot_date`, and source-specific metadata. This provenance is used for commercial-safe filtering at any pipeline stage.

## Key Constraints

- **Never install flash-attn** — conflicts with flashinfer on Blackwell
- **Unsloth may not support Qwen3_5 yet** — PEFT fallback is primary path; `Unsloth unavailable` warning is expected
- **OOM during CPT**: reduce `per_device_train_batch_size` to 1, double `gradient_accumulation_steps`
- **PEFT `target_modules` error**: inspect layer names with `print([n for n,_ in model.named_modules()])` — Qwen3.6 layer names differ from standard transformers
- **GGUF garbled output**: chat template mismatch — verify `<|im_start|>` tokens in GGUF tokenizer config
- **lm-eval task not found**: run `lm-eval --tasks list | grep -i pl` for current Polish task IDs

## English Retention Monitoring

Retention is measured **out-of-band**, not inside the trainer. There is no in-training English
benchmark: `english_retention_check` is commented out in `configs/cpt.yaml` and nothing in
`scripts/train/` reads it. To check for catastrophic forgetting, evaluate a checkpoint against
the base model:

```bash
python scripts/eval/run_eval.py --peft models/cpt/checkpoint-2000 \
    --base-model Qwen/Qwen3.6-27B --suite english
```

If English scores drop >3% below baseline, rebuild the CPT mix with a higher replay fraction —
`build_cpt_mix.py --replay-fraction 0.22` (it is a CLI flag, not a constant in the script; the
`replay_fraction` key in `configs/cpt.yaml` is documentation only and is read by nothing).