# Polish LLM Fine-Tuning Pipeline

[![ci](https://github.com/pswitala/pslab-finetuning/actions/workflows/ci.yml/badge.svg)](https://github.com/pswitala/pslab-finetuning/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

End-to-end QLoRA fine-tuning pipeline targeting Polish language fluency, factual
knowledge from Polish open-data catalogs baked into model weights, and **agentic
tool-use** (function calling grounded in the real GUS / dane.gov.pl / ISAP APIs). Final
artifact: **GGUF** files runnable in llama.cpp / Ollama.

**Default model:** Qwen3.6-27B (SSM-hybrid VLM, 96 GB VRAM target). To use a different
model, set `base_model:` in `configs/cpt.yaml` (and `configs/sft.yaml` / `configs/dpo.yaml`)
to any HuggingFace model ID or local path. The pipeline auto-detects LoRA target layers
(including fused-QKV architectures like Phi-3, and it fails loudly rather than silently
mis-targeting) and freezes vision encoders only when present. See `configs/models/` for
per-architecture presets (Llama-3, Phi-3, Mistral).

**Hardware:** 4× NVIDIA RTX 6000 Pro Blackwell, 96 GB VRAM each (QLoRA; runs on one card,
scales to all four with DDP — see [Multi-GPU training](#multi-gpu-training)).

> ### ⚠ Status: pilot-validated, not production-run
>
> The pipeline is complete and has been validated end-to-end by a short pilot. **No full
> production training run has been done yet**, and the shipped configs are deliberately
> **smoke-test sized**:
>
> | Config | Ships as | Meaning |
> |---|---|---|
> | `configs/cpt.yaml` | `max_steps: 20`, `bs=1 × ga=4`, `max_seq_len: 2048` | 20 optimizer steps — a stack-health check, not a model |
> | `configs/sft.yaml` | `max_steps: 200`, `bs=2 × ga=16` | 200 steps, not the 3 epochs the recipe calls for |
> | `configs/dpo.yaml` | `bs=4 × ga=8`, no `max_steps` | already full-run sized |
>
> `max_steps` overrides `num_train_epochs`, so running these as-is will **not** produce a
> usable model. See [Full-run configuration](#full-run-configuration) before committing GPU
> time. Also read [Known limitations](#known-limitations) — a few documented features are not
> yet wired in.

---

## Quickstart

The shortest path to proving the stack works on your machine. This is a **smoke test**, not a
training run — it produces a throwaway 20-step adapter. Follow the
[full walkthrough](#step-0--environment) for a real model.

```bash
# 0. Environment (see docs/SETUP.md for the full story)
pip install -r requirements.txt
python scripts/check_env.py          # expect sm_120, ~96 GB, bf16 OK

# 1. Smallest useful corpus: dane.gov.pl descriptions (minutes, no GPU)
#    Written to a scratch dir — the real pipeline splits data/catalogs/ first (Step 1a).
python scripts/ingest/dane_gov.py --out data/scratch/datasets.jsonl \
    --max-pages 5 --commercial-safe

# 2. Build a tiny CPT mixture from it
python scripts/process/build_cpt_mix.py \
    --pl "data/scratch/*.jsonl" --out data/processed/cpt --commercial-safe

# 3. 20-step CPT (configs/cpt.yaml already ships with max_steps: 20)
python scripts/train/cpt.py --config configs/cpt.yaml
# Watch: loss decreasing, no CUDA errors, no OOM, a checkpoint written

# 4. Confirm the base model still evaluates (adapter path, no merge needed)
python scripts/eval/run_eval.py --peft models/cpt --suite polish_quick \
    --base-model Qwen/Qwen3.6-27B --limit 50
```

If all four steps pass, the stack is healthy. Now go to
[Full-run configuration](#full-run-configuration).

---

## Pilot results

From the 200-step CPT pilot on this machine (see [Step 5](#step-5--pilot-run) for the full
metrics and how to read them):

| Task | Metric | Baseline (`Qwen/Qwen3.6-27B`) | CPT pilot (200 steps) | Change |
|---|---|---|---|---|
| `belebele_pol_Latn` | acc | 93.0% | 92.5% | −0.5% |
| `arc_challenge_mt_pl` | acc_norm | 52.0% | 50.5% | −1.5% |

**These numbers are not a result — they are a health check.** 200 steps is a tiny fraction of
a full CPT run, and benchmark gains only appear after the model has seen enough Polish text to
shift its representations. A pilot showing no change or a small dip is expected and correct.
The pilot's job is to prove the training stack works, not to show improved Polish scores.

---

## Table of contents

**Walkthrough**
- [Step 0 — Environment](#step-0--environment)
- [Step 1 — Download open-data catalogs](#step-1--download-open-data-catalogs)
- [Step 2 — Download pre-training corpora](#step-2--download-pre-training-corpora)
- [Step 3 — Clean and deduplicate the corpus](#step-3--clean-and-deduplicate-the-corpus)
- [Step 4 — Build training datasets](#step-4--build-training-datasets)
- [Step 5 — Pilot run](#step-5--pilot-run)
- [Step 6 — Continued Pretraining (CPT)](#step-6--continued-pretraining-cpt)
- [Step 7 — Supervised Fine-Tuning (SFT)](#step-7--supervised-fine-tuning-sft)
- [Step 8 — Preference Optimization (DPO)](#step-8--preference-optimization-dpo)
- [Step 9 — Evaluate](#step-9--evaluate)
- [Step 10 — Export to GGUF](#step-10--export-to-gguf)

**Reference**
- [Understanding the core training knobs](#understanding-the-core-training-knobs)
- [Full-run configuration](#full-run-configuration)
- [Multi-GPU training](#multi-gpu-training)
- [Checkpointing — saving, resuming, and stopping safely](#checkpointing--saving-resuming-and-stopping-safely)

**Project**
- [Development](#development)
- [Quick reference — all commands in order](#quick-reference--all-commands-in-order)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Step 0 — Environment

**Prerequisites:** Python 3.11+, an NVIDIA GPU with `sm_120` (Blackwell) or adjust the
thresholds below, a Hugging Face account with a token, and — for Step 3a — the fastText
language-ID model. Multi-GPU additionally requires Linux (NCCL has no Windows build).

**→ Full environment setup, including the pinned package inventory, the fastText and KenLM
downloads, the llama.cpp build, and a Blackwell-specific troubleshooting table, lives in
[docs/SETUP.md](docs/SETUP.md).** This section is the short version.

```bash
# Install into the existing vllm virtualenv — torch, vllm and flashinfer are already
# present there, so this only pulls the missing training packages.
pip install -r requirements.txt

# Hugging Face authentication (needed for gated corpora like CulturaX)
huggingface-cli login

# Copy and fill in .env  (HF_TOKEN, optionally WANDB_API_KEY / WANDB_PROJECT)
cp .env.example .env
```

If you are building a **fresh** environment rather than reusing the vllm venv, two packages
are not on the default PyPI index and need their own wheel servers:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://flashinfer.ai/whl/cu128/torch2.10/
```

**Why two custom wheel indices:** `torch==2.10.0` with CUDA 12.8 / `sm_120` (Blackwell)
support lives on the PyTorch wheel server, and `flashinfer-python==0.6.6` with Blackwell
kernels lives on the FlashInfer wheel server. Passing both as `--extra-index-url` lets pip
resolve them alongside the rest of the packages from PyPI in a single pass. The pinned
`nvidia-*-cu12==12.8.x` runtime wheels in `requirements.txt` are what make **cu128** the
correct index — a cu129 build will fight those pins.

**Why not flash-attn:** `flashinfer-python==0.6.6` is the Blackwell-compatible attention
backend. `flash-attn` does not support `sm_120` and conflicts with flashinfer if both are
installed — do not install it.

**Why bitsandbytes for quantization:** bitsandbytes is the standard 4-bit NF4 training
backend. Alternatives like GPTQ and AWQ produce inference-only quantized weights — you
cannot attach trainable LoRA adapters on top of them. Only bitsandbytes keeps the
quantized base frozen while training full-precision LoRA adapters alongside it.

`requirements.txt` also pins `wandb` (all three training stages log to Weights & Biases
when `WANDB_API_KEY` is set) and documents the two optional special-install backends
(`unsloth`, `llama-cpp-python`) in a trailing comment block.

Verify the GPU and stack before doing anything else:

```bash
python scripts/check_env.py
# Expected: RTX 6000 Pro Blackwell, sm_120, ~96 GB, bf16 matmul OK
# Also reports vllm / llama_cpp availability, and NCCL when >1 GPU is visible.

# On a non-reference GPU, relax the warning thresholds:
python scripts/check_env.py --min-vram 24 --min-compute 8.0

# Before a multi-GPU run, require all four cards:
python scripts/check_env.py --min-gpus 4
```

---

## Step 1 — Download open-data catalogs

**What it does:** Harvests Polish factual text from three government sources and stores each
record with provenance metadata (`source`, `license`, `snapshot_date`). This text becomes two
inputs: raw training signal for CPT and the ground truth for synthetic Q&A in SFT.

**Why catalog knowledge is injected via training (not RAG):** The goal is a model that answers
questions about Polish law and statistics without needing any retrieval system at inference
time. RAG is simpler to add but creates a deployment dependency and fails silently when the
retrieval index is stale or wrong. Baking facts into weights is harder but produces a fully
self-contained model, which is a requirement for the GGUF/Ollama deployment target.

**Why these three sources:**
- **Sejm/ISAP** — the official Polish legal journal. Acts from Dziennik Ustaw (DU) and Monitor Polski (MP) are public-domain, authoritative, and written in formal Polish — high-signal training text.
- **dane.gov.pl** — the national open-data portal. Dataset descriptions are concise Polish prose covering diverse domains (health, agriculture, environment) and carry explicit CC-BY / CC0 licenses, so they are safe for commercial use.
- **GUS BDL** — Statistics Poland's statistical database. ~5.5 M indicator-value pairs covering demographics, economy, and infrastructure from 2010 onward. Public-domain and verbalized as Polish sentences — dense factual coverage that CPT and catalog Q&A both benefit from.

**Why provenance per record:** The `license` field on every record lets `build_cpt_mix.py` run
`--commercial-safe` filtering at mix time without re-downloading data. The `snapshot_date`
defines the model's knowledge cutoff precisely, which is important to communicate to end users.

```bash
# Legal acts from Sejm/ISAP — public domain, strong CPT + SFT signal
python scripts/ingest/sejm_isap.py \
    --publisher DU --years 2015-2024 \
    --out data/catalogs/isap/du_2015_2024.jsonl

python scripts/ingest/sejm_isap.py \
    --publisher MP --years 2015-2024 \
    --out data/catalogs/isap/mp_2015_2024.jsonl

# dane.gov.pl — national open-data portal descriptions (CC-BY / CC0)
python scripts/ingest/dane_gov.py \
    --out data/catalogs/dane_gov/datasets.jsonl \
    --max-pages 200 --commercial-safe

# GUS BDL — statistical indicators (public domain), verbalized as Polish sentences
# Subject IDs are alphanumeric (K11, K15, K27 …) — discover them first:
python scripts/ingest/gus_bdl.py --list-subjects

# Then harvest selected subjects (pick from the printed list):
# --max-vars-per-subject caps variables per subject (K11 alone has 7700+, causes 429s)
# --delay sets seconds between requests (default 0.5; increase if you still get 429s)
python scripts/ingest/gus_bdl.py \
    --subjects K11,K15,K27,K43,K47,K44,K23,K24,K54,K3,K9,K20,K21,K8,K10,K22 \
    --years 2010-2025 \
    --max-vars-per-subject 300 \
    --delay 0.6 \
    --out data/catalogs/gus_bdl/indicators.jsonl
```

### 1a. Carve out the evaluation holdout — do this now

**Do this immediately after ingest and before building any dataset.** The closed-book catalog
eval in [Step 9](#step-9--evaluate) is the most direct test of this project's core goal
(knowledge injection without RAG), and it is only meaningful if the questions come from
records the model never saw. `make_holdout.py` splits the catalogs reproducibly using a stable
per-id hash, so the same records land in the holdout on every run.

```bash
python scripts/process/make_holdout.py \
    --input "data/catalogs/**/*.jsonl" \
    --train-out data/catalogs_train \
    --holdout-out data/catalogs_holdout \
    --fraction 0.02
```

> **Everything downstream must read `data/catalogs_train/`, never `data/catalogs/`.**
> `data/catalogs/` still contains the *unsplit* records, so a glob over it silently
> re-includes the holdout and invalidates your eval scores. The holdout is written to
> `data/catalogs_holdout/` — a *sibling* of `data/catalogs/`, deliberately not nested inside
> it, so no `data/catalogs/**` glob can reach it by accident.
>
> Alternatively, both builders accept `--exclude-ids <file>` to filter by record id instead of
> by path, which is useful if you'd rather keep one catalog directory.

---

## Step 2 — Download pre-training corpora

**What it does:** Downloads large Polish text corpora from Hugging Face for general language modeling signal, plus Polish instruction and preference datasets for the SFT and DPO stages, and an English corpus for replay.

**Why each corpus was chosen:**

| Corpus | Size | License | Role | Why this, not another |
|---|---|---|---|---|
| **HPLT 2.0 Polish** | ~400 GB | CC0 | CPT general text | Most permissive license possible; deduplicated web crawl at scale; FLORES-200 language IDs are reliable |
| **Polish Wikipedia** | ~1 GB | CC-BY-SA | CPT general text | Clean, encyclopedic, factual — low noise, high quality |
| **CulturaX Polish** | ~150 GB | ODC-BY | CPT general text | Cleaned web text; good topical breadth; complements HPLT |
| **Dolci-Instruct SFT** | ~495k | Apache 2.0 | SFT instructions | High-quality Polish translations of diverse instructions; Apache 2.0 enables commercial use |
| **Dolci-Instruct DPO** | ~225k | Apache 2.0 | DPO preferences | Matching preference pairs with chosen/rejected for the same prompts |
| **English replay (C4)** | ~350 GB | CC-BY | CPT anti-forgetting | Broad web text; CC-BY is commercial-safe; 1024 shards — download only what you need for 18% target |

**Why English replay:** Qwen3.6-27B has only an instruct checkpoint — no base model was published. CPT on Polish text alone, starting from an instruct model, risks catastrophic forgetting of English and general reasoning. Interleaving ~18% English text during CPT keeps the model's English capabilities stable without wasting most of the compute budget on a language it already knows well.

**Why FLORES-200 language codes:** CulturaX and HPLT use FLORES-200 codes (`pol_Latn`) rather than ISO 639-1 (`pl`). Passing the wrong code either returns an error or, worse, silently downloads a different language split.

```bash
# Recommended download approach using the HF datasets CLI / Python:
python - <<'EOF'
from datasets import load_dataset

# HPLT 2.0 Polish — CC0, 400 GB, best license
# Uses FLORES-200 language codes: Polish = "pol_Latn" (not "pl")
ds = load_dataset("HPLT/HPLT2.0_cleaned", "pol_Latn", split="train")
ds.to_json("data/raw/hplt_pl/data.jsonl", lines=True, force_ascii=False)

# Polish Wikipedia
ds = load_dataset("wikimedia/wikipedia", "20231101.pl", split="train")
ds.to_json("data/raw/wikipedia_pl/data.jsonl", lines=True, force_ascii=False)

# CulturaX Polish — ODC-BY, ~150 GB
# NOTE: CulturaX uses FLORES-200 language codes, not ISO 639-1.
#       Polish = "pol_Latn" (not "pl").
ds = load_dataset("uonlp/CulturaX", "pol_Latn", split="train")
ds.to_json("data/raw/culturax_pl/data.jsonl", lines=True, force_ascii=False)
EOF

# OpenEuroLLM instruction + DPO datasets for SFT/DPO stages
# (SpeakLeash training data is NOT publicly released on HuggingFace.)
# These are Apache 2.0, commercial-safe, ~495k SFT + ~225k DPO Polish examples.
python - <<'EOF'
from datasets import load_dataset
import os

os.makedirs("data/raw/dolci-sft-pl", exist_ok=True)
os.makedirs("data/raw/dolci-dpo-pl", exist_ok=True)

# SFT: messages format (role/content) — filter to Polish subset
ds = load_dataset("openeurollm/Dolci-Instruct-SFT-translated", name="pl", split="train")
ds.to_json("data/raw/dolci-sft-pl/data.jsonl", lines=True, force_ascii=False)

# DPO: prompt/chosen/rejected format — filter to Polish subset
ds = load_dataset("openeurollm/Dolci-Instruct-DPO-translated", name="pl", split="train")
ds.to_json("data/raw/dolci-dpo-pl/data.jsonl", lines=True, force_ascii=False)
EOF
```

For large corpora that don't fit in memory, or when disk space is tight, convert parquet shards one at a time using the shard-by-shard scripts — they append to JSONL as each shard is processed and delete the cached parquet afterward:

```bash
# CulturaX — gated, run huggingface-cli login first; 160 shards (~1.96 GB each)
python scripts/ingest/culturax_pl.py --out-dir data/raw/culturax_pl
python scripts/ingest/culturax_pl.py --out-dir data/raw/culturax_pl --shards 3  # smoke test

# HPLT 2.0 Polish — not gated; 1578 shards (~345 MB each)
python scripts/ingest/hplt_pl.py --out-dir data/raw/hplt_pl
python scripts/ingest/hplt_pl.py --out-dir data/raw/hplt_pl --shards 3  # smoke test

# English replay (C4) — not gated; 1024 shards (~345 MB each)
# 10 shards ≈ 3.5M docs, which covers the 18% replay target for most corpus sizes
python scripts/ingest/replay_en.py --out-dir data/raw/replay_en --shards 10
python scripts/ingest/replay_en.py --out-dir data/raw/replay_en  # all 1024 shards if disk allows
```

---

## Step 3 — Clean and deduplicate the corpus

**What it does:** Removes low-quality documents, non-Polish text, and near-duplicate content across all corpora before any training data is assembled.

### 3a — Quality filtering (`pipeline.py`)

**What it does:** Passes every document through four sequential filters: language detection, repetition detection, general quality heuristics, and Polish-specific heuristics.

> **Prerequisite:** this stage needs the fastText language-ID model at
> `data/models/lid.176.bin`. See [docs/SETUP.md](docs/SETUP.md) for the download command.

**Filter stages and why each was chosen:**

1. **fastText language-ID (`LanguageFilter`):** Classifies each document's language and drops anything below `--lang-threshold 0.7` confidence for Polish. Alternatives like `langdetect` are slower and less reliable for short documents. Rule-based approaches (character frequency, stopwords) work but require careful tuning — fastText has been trained specifically for this task. Using it as the first filter means no compute is wasted on clearly non-Polish text in later stages.

2. **Gopher Repetition Filter:** Removes documents with a high fraction of repeated lines or repeated n-grams. This is common in web crawls: boilerplate navigation text, legal disclaimers copy-pasted across pages, and table-of-contents pages all pass language detection but are terrible training signal. The Gopher paper (DeepMind, 2021) established these thresholds empirically on large web corpora.

3. **Gopher Quality Filter:** Applies word count bounds, minimum mean word length, and maximum symbol ratio. Documents shorter than 50 words rarely contain enough context for meaningful language modeling. Extremely long documents (>100k words) are usually structured data dumps disguised as text. High symbol ratios indicate code listings, spreadsheet exports, or OCR artifacts.

4. **Polish quality heuristics (`quality_pl.py`):** Adds two checks that standard Gopher misses for Polish specifically:
   - **Diacritic ratio:** Authentic Polish prose contains a characteristic frequency of ą, ę, ó, ś, ż, ź, ć, ń, ł. Text with near-zero diacritics (threshold: 0.008) is almost certainly not genuine Polish — it's either a different language that passed language detection, or OCR/encoding-corrupted text where diacritics became question marks.
   - **Stopword ratio:** Polish function words (i, w, na, z, że…) should appear regularly in natural prose. Texts with very few stopwords tend to be keyword lists, tables, or other non-prose content.

**Why datatrove:** datatrove parallelizes filtering across multiple workers with a clean pipeline abstraction. It handles the shard-level work distribution that would otherwise require manual multiprocessing. Alternative: write a plain Python loop with multiprocessing, but then you'd be reimplementing what datatrove already handles well.

```bash
# 3a. Quality filter + language-ID (runs on CPU, no GPU needed)
python scripts/process/pipeline.py \
    --input "data/raw/**/*.jsonl" \
    --output data/interim/clean \
    --workers 16 \
    --lang-threshold 0.7
```

### 3b — Cross-corpus MinHash deduplication (`dedup.py`)

**What it does:** Finds and removes near-duplicate documents across all corpora together, using MinHash LSH with a Jaccard similarity threshold of 0.8.

**Why cross-corpus dedup matters:** The same Wikipedia article frequently appears verbatim or near-verbatim in web crawls (HPLT, CulturaX). A legal act published in the Dziennik Ustaw is often reproduced word-for-word on government news sites. Deduplicating within each corpus separately would miss all of these cross-corpus duplicates. Training on the same text multiple times without intent skews the model toward memorizing high-frequency repeated content.

**Why MinHash (not exact dedup):** Exact hash-based dedup catches identical documents but misses near-duplicates — the same article with a timestamp added, or a legal act with one changed article number. MinHash estimates Jaccard similarity between document shingles and is O(n) in the number of documents via Locality Sensitive Hashing (LSH), making it practical for hundreds of GB.

**Why the 4-stage datatrove pipeline:** MinHash dedup cannot run in a single pass because the matching step requires global state (all signatures must be visible to find pairs). The four stages — signatures, buckets, clusters, filter — separate naturally parallelizable work (signatures, filter) from inherently sequential work (clustering). Running stages 1 and 4 across 16 workers while stage 3 uses 1 worker matches the actual data dependencies.

**Why threshold 0.8:** A Jaccard threshold of 0.8 means two documents share 80% of their shingles before being considered duplicates. Lower thresholds (0.6) incorrectly remove topically similar but genuinely distinct documents (e.g., two news articles about the same event). Higher thresholds (0.95) miss many real duplicates that differ only by minor boilerplate. 0.8 is the standard threshold used by the RefinedWeb and Dolma projects.

```bash
# 3b. Cross-corpus MinHash near-dedup (removes duplicates across all sources)
python scripts/process/dedup.py \
    --input data/interim/clean \
    --output data/interim/dedup \
    --workdir data/interim/_minhash \
    --workers 16 \
    --threshold 0.8
```

`--threshold` actually drives the LSH configuration: the target Jaccard similarity is
converted to a MinHash band count (`num_buckets ≈ threshold^(-hashes_per_bucket)`) and
passed into `MinhashConfig` (earlier versions accepted the flag but ignored it, silently
using datatrove defaults). Tune the curve with `--num-permutations` (default 112) and
`--n-grams` (default 5) if needed.

---

## Step 4 — Build training datasets

> All four builders read **`data/catalogs_train/`**, the training split produced by
> [Step 1a](#1a-carve-out-the-evaluation-holdout--do-this-now). Do not point them at
> `data/catalogs/`.

### 4a. CPT mixture (`build_cpt_mix.py`)

**What it does:** Combines cleaned Polish documents with a controlled fraction of English text and writes the result as sharded parquet files ready for the trainer.

**Why parquet (not JSONL):** The training loop loads data via HuggingFace `datasets`, which reads parquet efficiently via Apache Arrow's columnar format. Random access into a 400 GB JSONL file is slow; parquet partitioned into 100k-row shards can be memory-mapped and accessed at the row level. This matters because the trainer needs to shuffle across the entire dataset during each epoch.

**Why 18% English replay:** The replay fraction was set to balance two competing risks. Too little English (<10%) and the model forgets English reasoning chains and MMLU-style tasks after one CPT epoch. Too much English (>25%) wastes training compute on a language the model already knows well, reducing the effective Polish exposure per GPU-hour. The 15–20% range is consistent with what multilingual CPT papers (GlotLID, EMMA-500) report as effective for adding a new language while preserving existing ones.

**Why no tokenization at this stage:** The trainer uses `packing=True` during CPT, which concatenates documents with a separator token and packs them into full-length sequences. This is more efficient than pre-tokenizing: you avoid padding waste and the trainer can rebalance sequence lengths across batches. Pre-tokenizing would also couple the dataset to a specific `max_seq_len`, making it harder to experiment with different context lengths.

**Why streaming (not an in-memory shuffle):** The builder never loads the corpus into RAM. It makes two lightweight counting passes (to size the English replay), then a single writing pass that assigns each document to a random parquet shard and Bernoulli-samples the English stream down to the target fraction. Random shard assignment gives an approximate global shuffle — so Polish and English are interleaved at the document level and each batch sees a mixture — while scaling to the 400 GB+ corpus that an in-memory `shuffle()` could never hold. (The trainer also shuffles per epoch on top of this.)

**Why `--max-per-source` (balancing a dominant source):** Sources differ enormously in count
and length. GUS BDL alone contributes ~5.5 M short (~130-token) statistic sentences — that
can be **half the total token budget**, drowning out the longer web/legal prose that actually
drives *fluency*, and inflating the step count of the run.

`--max-per-source SOURCE=N` uniformly random-subsamples a source down to `N` documents, using
the same Bernoulli trick as the English replay, so it doesn't bias toward whichever shard
streamed first. Capping GUS both **shortens the run** (fewer tokens → fewer steps) and
**rebalances toward prose**. The English replay is sized off the *post-cap* Polish count, so
the 18% ratio stays correct.

Example: `--max-per-source gus_bdl=1000000` keeps ~1 M of 5.5 M GUS records, roughly halving
the total token count and therefore roughly halving the number of optimizer steps. (Token and
step *ratios* like this are rig-independent; for wall-clock, see
[projecting run length](#projecting-run-length).)

Inspect the mix by source with:

```python
import pyarrow.dataset as ds, collections
d = ds.dataset("data/processed/cpt/train", format="parquet"); c = collections.Counter()
for b in d.to_batches(columns=["domain","source"]):
    for dom, src in zip(b.column("domain").to_pylist(), b.column("source").to_pylist()): c[(dom,src)] += 1
print(c.most_common())
```

```bash
python scripts/process/build_cpt_mix.py \
    --pl "data/interim/dedup/**/*.jsonl*" \
         "data/catalogs_train/**/*.jsonl" \
    --en "data/raw/replay_en/**/*.jsonl" \
    --out data/processed/cpt \
    --replay-fraction 0.18 \
    --max-per-source gus_bdl=1000000 \
    --commercial-safe
# NOTE: dedup output is gzipped (*.jsonl.gz) — use *.jsonl* so the glob matches it,
# otherwise the entire deduped web corpus is silently skipped.
# --max-per-source SOURCE=N caps a source (uniform random subsample); repeatable, e.g.
#   --max-per-source gus_bdl=1000000 c4=800000.  Uncapped sources are unaffected.
# --exclude-ids FILE drops records by id — an alternative to the catalogs_train split.
# Output: data/processed/cpt/train/*.parquet  +  .../val/*.parquet
```

### 4b. SFT instruction dataset (`build_sft_qa.py`)

**What it does:** Generates synthetic Polish Q&A pairs from catalog records and merges them with the downloaded Dolci-Instruct dataset. Each catalog record produces up to 2 question-answer pairs using source-specific templates (different templates for legal acts vs. GUS statistics vs. dane.gov.pl).

**Why template-based QA (not LLM-generated):** Three reasons. First, license safety — questions generated by a commercial LLM inherit that LLM's terms of service, which may restrict use. Template-generated questions are entirely original. Second, controllability — templates ensure every Q&A pair is strictly grounded in the source record; LLM-generated answers sometimes add facts not in the source. Third, no dependency — running a teacher LLM for millions of records would cost significant money and add infrastructure complexity. The template approach is deterministic, free, and reproducible.

> `--mode` also accepts `llm`, which generates pairs with a teacher model instead of
> templates. It exists for experimentation and inherits the license and grounding caveats
> above; `template` is the supported default.

**Why per-record 2:** One question per record is a wasted opportunity given the diversity of question templates. More than 2 risks the SFT dataset becoming dominated by formulaic catalog Q&A, which could hurt general instruction following. Two pairs per record gives coverage diversity while keeping the catalog fraction of the SFT mix reasonable relative to the 495k Dolci-Instruct examples.

**Why cat rather than a merge script:** The Dolci-Instruct data is already in chat format; the catalog Q&A output is also chat format. Shell `cat` is sufficient for the merge, and the val split is taken from the merged file. If the format ever changes, replace the cat with a Python merge script.

```bash
# Generate synthetic Q&A from catalog records (training split only)
python scripts/process/build_sft_qa.py \
    --input "data/catalogs_train/**/*.jsonl" \
    --out data/processed/sft/catalog_qa.jsonl \
    --mode template --per-record 2

# Merge with downloaded instruction datasets (+ agentic data from 4c below)
cat data/raw/dolci-sft-pl/data.jsonl \
    data/processed/sft/catalog_qa.jsonl \
    data/processed/sft/agentic/tool_qa.jsonl \
    > data/processed/sft/train.jsonl
# Create a small val split (e.g. first 1000 lines)
head -1000 data/processed/sft/train.jsonl > data/processed/sft/val.jsonl
```

### 4c. Agentic (tool-use) dataset (`build_sft_qa.py --mode agentic`)

**What it does:** Generates deterministic tool-use trajectories grounded in the *same* APIs
the catalog records came from. Each sample is a full function-calling conversation — user
question → assistant `tool_calls` → `role:"tool"` result → final grounded answer — plus the
`tools` schema. This is what teaches the model to *call tools*, the third project goal
alongside Polish fluency and knowledge injection.

**Why grounded in real APIs (not hallucinated tools):** `scripts/common/tool_catalog.py`
defines three function schemas (`gus_bdl_query`, `dane_gov_search`, `isap_lookup`) that mirror
the GUS BDL / dane.gov.pl / ISAP endpoints already scraped in Step 1. The arguments in each
training sample are filled from the record's real `meta` (subject/variable IDs, publisher/year/
position, dataset titles), so the model learns argument shapes that map onto calls that
actually exist. Every generated sample is validated against the tool's JSON Schema
(`scripts/common/tooling.py`) and dropped if it doesn't conform. All three tools are offered on
every sample, so the model also learns tool *selection*, not just argument filling.

```bash
python scripts/process/build_sft_qa.py \
    --input "data/catalogs_train/**/*.jsonl" \
    --out data/processed/sft/agentic/tool_qa.jsonl \
    --mode agentic --per-record 2
```

### 4d. DPO preference dataset (`build_dpo.py`)

**What it does:** Validates and filters the Dolci-DPO pairs instead of copying them verbatim.
The raw dump contains malformed pairs (empty responses, `chosen == rejected`, exact
duplicates) that add no preference signal or actively hurt training; `build_dpo.py` drops
those and writes a clean train/val split that `scripts/train/dpo.py` consumes directly.

```bash
python scripts/process/build_dpo.py \
    --input data/raw/dolci-dpo-pl/data.jsonl \
    --out data/processed/dpo \
    --val 500
# Output: data/processed/dpo/train.jsonl + val.jsonl
```

---

## Step 5 — Pilot run

**What it does:** Runs a short training run through the full CPT pipeline — model loading,
4-bit quantization, LoRA attachment, forward pass, gradient computation, checkpoint save —
and exits cleanly.

**Why run a pilot before committing to multi-day training:** Blackwell's `sm_120` compute capability is new enough that some CUDA kernel paths have subtle incompatibilities. A short pilot catches: OOM from incorrect batch size estimates, PEFT `target_modules` errors if layer names differ from what the config expects, bitsandbytes CUDA errors from version mismatches, and gradient overflow with bf16. Finding any of these a few hundred steps in takes minutes; finding them 12 hours into a CPT run means restarting from scratch.

**How many steps:** enough to see loss decrease monotonically (confirming gradients flow
correctly) and to trigger at least one checkpoint save (confirming disk writes work), but not
so many that a failed pilot wastes GPU time. `configs/cpt.yaml` ships `max_steps: 20`, which
is the minimum useful smoke test; raise it to ~200 for the fuller pilot whose metrics are
reported below.

```bash
# configs/cpt.yaml already ships pilot-sized:
#   max_steps: 20, per_device_train_batch_size: 1, gradient_accumulation_steps: 4,
#   max_seq_len: 2048
# Raise max_steps to 200 to reproduce the metrics below.

python scripts/train/cpt.py --config configs/cpt.yaml
# Watch: loss should decrease, no CUDA errors, no OOM
```

If you hit OOM, reduce `per_device_train_batch_size` and increase
`gradient_accumulation_steps` to compensate (keep effective batch size the same). See
[Understanding the core training knobs](#understanding-the-core-training-knobs).

### CPT pilot — training metrics

Observed on a 200-step pilot (RTX 6000 Pro Blackwell, QLoRA r=64, packing, and the shipped
pilot geometry `per_device_train_batch_size=1 × gradient_accumulation_steps=4 ×
max_seq_len=2048` — an effective batch of 4): **33.24 s/optimizer step**, ~5.3 h wall-clock
including periodic eval.

Train vs. eval comparison at the end of the pilot:

| Metric | Train (avg, 200 steps) | Eval (final checkpoint) |
|---|---|---|
| Loss | 1.543 | 1.304 |
| Token accuracy | ~0.82 | 0.705 |
| Entropy | ~0.70 | 1.308 |

**Signal quality reference — weak / good / great:**

| Signal | Weak | Good | Great | Pilot observed |
|---|---|---|---|---|
| Loss | > 2.5 | 1.3 – 2.0 | < 1.3 | 1.543 train / 1.304 eval ✓ |
| Token accuracy | < 0.60 | 0.65 – 0.80 | > 0.80 | 0.82 train / 0.705 eval ✓ |
| Entropy | > 2.5 | 0.8 – 2.0 | < 0.8 | 0.70 train / 1.308 eval ✓ |
| Grad norm | > 20 or NaN | 1 – 8 | 0.5 – 3 | stable throughout ✓ |
| Step time (RTX 6000 Pro) | > 100 s | 30 – 60 s | < 30 s | 33 s ✓ |

Key notes:
- **Train/eval gap (accuracy 0.82 → 0.705, entropy 0.70 → 1.308) is expected** at 200 steps — the model is more confident on sequences it has seen; this is normal generalization behaviour, not overfitting.
- **Step time > 100 s** means `flash-linear-attention` / `causal-conv1d` kernels are missing and the SSM layers fall back to PyTorch (~10× slower). Install with `pip install flash-linear-attention causal-conv1d`.
- **Loss > 2.5 at step 1 is normal** — it should fall below 2.0 within the first 20–50 steps as the model starts fitting the new distribution.
- **NaN loss at any step**: stop immediately — likely bf16 overflow or a malformed batch. Check the last clean `logging_steps` entry.

### Projecting run length

Wall-clock depends on your corpus, your config, and your GPU count, so compute it rather than
copying a number:

```
days ≈ (optimizer_steps × seconds_per_step) / 86 400
```

Get `optimizer_steps` for *your* corpus and config — it is not a property of the model — with:

```bash
python scripts/train/count_steps.py --config configs/cpt.yaml --num-gpus 4
```

Two things make naive projections badly wrong:

- **Step count scales inversely with effective batch.** The pilot geometry has an effective
  batch of 4; the [full-run configuration](#full-run-configuration) has 32. That is ~8× fewer
  optimizer steps over the same data — at a higher cost per step. Never multiply the pilot's
  step count by the pilot's step time and assume it describes a full run.
- **Eval overhead does not scale linearly** with training time; it is governed by `eval_steps`.

### Pilot eval — CPT results

After the pilot, run the quick Polish eval to confirm nothing is broken:

```bash
# With PEFT adapter (fast — base loads from HF cache):
python scripts/eval/run_eval.py --peft models/cpt --suite polish_quick \
    --base-model Qwen/Qwen3.6-27B

# Or after merging:
python scripts/eval/run_eval.py --model models/cpt/merged --suite polish_quick
```

Observed results are in [Pilot results](#pilot-results) at the top of this README. How to act
on them:

| Metric change (pilot scale) | Meaning | Action |
|---|---|---|
| ±2% on either benchmark | Noise — expected | Continue to full CPT |
| −3% to −5% on `acc_norm` | Early degradation signal | Monitor; may need higher replay fraction |
| > −5% drop, or NaN loss | Something is wrong | Stop — debug LR, replay data, or bf16 overflow |

---

## Step 6 — Continued Pretraining (CPT)

**What it does:** Trains the model on raw Polish text (web corpora, Wikipedia, legal acts, GUS statistics) plus English replay using next-token prediction on packed sequences. Produces a Polish-fluent model that still follows instructions. This is the most compute-intensive stage.

**Why start from the instruct checkpoint (Qwen default):** Qwen3.6-27B has no publicly released base checkpoint — only the instruct model exists. This is unusual but works, with adjustments. Starting from instruct means the model already knows how to follow instructions; CPT risks eroding this if done carelessly. (If your model has a base checkpoint, prefer it for CPT — use a standard LR of 1e-4 and reduce English replay accordingly.)

**Why conservative learning rate (3e-5 vs. typical 1e-4):** A high LR in CPT on an instruct model causes alignment degradation — the model stops following instructions and generates coherent Polish prose but ignores user intent. 3e-5 gives enough gradient signal to adapt to Polish text distribution while keeping the instruction-following behavior mostly intact. The English replay also helps anchor the model to its original capabilities.

**⚠ Match the LR to base vs. instruct — highest-impact CPT setting.** The 3e-5 default exists
only to *preserve instruct alignment*. If your `base_model` is a **base (non-instruct)
checkpoint** — e.g. `google/gemma-4-12B` rather than `google/gemma-4-12B-it` — there is no
alignment to protect, so 3e-5 is needlessly timid and, on a short single-epoch run over a
modest token budget, will **under-adapt** (the Polish shift barely takes).

For a base checkpoint use a **standard CPT LR of ~1e-4** (or a cautious `5e-5`); reserve
`3e-5` for continuing from an instruct model. Check which checkpoint you're starting from
*before* committing a multi-day run — this one value largely determines whether CPT visibly
moves the model. The same logic relaxes the English-replay concern: replay guards forgetting,
but catastrophic *alignment* loss isn't a risk when there's no instruct alignment to begin
with.

**Why QLoRA (not full fine-tuning):** At 27B parameters and bf16 precision, the model alone occupies ~54 GB of VRAM. Full fine-tuning requires optimizer states (Adam: 2× model size) on top — well over 96 GB. QLoRA loads the base model in 4-bit NF4 (~14 GB), keeps it frozen, and attaches trainable 16-bit LoRA adapters (~1–2 GB depending on rank). This fits comfortably within 96 GB while still adapting 27B parameters of representation capacity through the adapter layers.

**Why `target_modules: auto`:** LoRA only updates the projection layers you name. For a
pure-attention model (Llama, Mistral) the correct targets are `q_proj`, `k_proj`, `v_proj`,
`o_proj`, and the MLP projections — leaving those out means the adapter can only influence
25–50% of the model's representation capacity.

For SSM-hybrid models like Qwen3.6-27B, 75% of layers are linear-attention/SSM
(`linear_attn.*`) with different projection names (`in_proj_qkv`, `in_proj_z`, `in_proj_a`,
`in_proj_b`) — missing them effectively freezes 48 of 64 layers. `target_modules: auto`
inspects the loaded model's named modules and selects the right projections automatically,
including fused-QKV layouts (Phi-3's `qkv_proj` / `gate_up_proj`) via a best-match pattern. If
it detects nothing it raises with the candidate module names rather than silently falling back
to modules that may not exist. Override with an explicit list for exotic architectures (see
`configs/models/`).

**Why rsLoRA (`use_rslora: true`):** Standard LoRA scales the adapter output by `alpha/r`. At high rank (r=64), this scaling can cause gradient instability during the early warmup steps. rsLoRA replaces this with `alpha/sqrt(r)`, which remains stable across a wider range of ranks and makes it safe to use r=64 for CPT without careful per-run LR tuning.

**Why r=64 for CPT (higher than SFT's r=32):** CPT is adapting the model to a new language distribution — a larger distributional shift than instruction following. Higher rank gives more expressiveness in the adapter. The compute cost is acceptable at CPT's 1-epoch schedule; for SFT's 3-epoch schedule, the smaller r=32 is sufficient.

**Why packing=True:** CPT operates on raw text documents of varying length. Without packing, each training sequence would be padded to `max_seq_len`, wasting 30–60% of compute on padding tokens. Packing concatenates documents with a separator and fills sequences to capacity, achieving near-100% GPU utilization. This is only appropriate for CPT (where document boundaries don't affect the learning objective). SFT must not use packing because loss masking on responses would span document boundaries incorrectly.

**Monitoring English retention:** CPT on a monolingual Polish corpus on top of a multilingual
instruct model risks catastrophic forgetting. Retention is currently measured **out-of-band** —
evaluate a mid-run checkpoint against the base model and compare:

```bash
python scripts/eval/run_eval.py --peft models/cpt/checkpoint-2000 \
    --base-model Qwen/Qwen3.6-27B --suite english
```

If English accuracy drops more than ~3% below baseline, increase the replay fraction
(`--replay-fraction 0.22`), rebuild the CPT mix, and restart from the last clean checkpoint.
An in-trainer `english_retention_check` is sketched in `configs/cpt.yaml` but is **not yet
wired into `cpt.py`** — see [Known limitations](#known-limitations).

```bash
python scripts/train/cpt.py --config configs/cpt.yaml
# With the shipped config this runs 20 steps (a smoke test).
# For a real run see Full-run configuration — remove max_steps so num_train_epochs: 1 governs.
# Checkpoints saved to models/cpt/ every 500 steps (save_steps).

# When training finishes, merge the adapter into full weights:
python scripts/train/cpt.py --config configs/cpt.yaml --merge
# Output: models/cpt/merged/
```

---

## Step 7 — Supervised Fine-Tuning (SFT)

**What it does:** Teaches the model to follow Polish instructions and answer questions about catalog knowledge in a conversational format. Starts from the CPT-merged weights, trains on chat-formatted examples, and computes loss only on assistant responses.

**Why start from CPT merged (not the original instruct model again):** CPT gave the model Polish fluency and catalog knowledge. Starting SFT from CPT/merged means the instruction-following layer is built on top of a model that already understands Polish — a significantly better initialization than going from the English-first instruct model directly to Polish instruction data. The downside is the two-stage pipeline takes more time, but the quality improvement is substantial for low-resource language adaptation.

**Why train_on_responses_only=True:** In standard causal language modeling, loss is computed over all tokens including user turns. This is correct for CPT (where there is no distinction between "input" and "output") but wrong for SFT. Learning to predict user messages teaches the model the distribution of user questions, which doesn't help it answer them. Masking user turns to zero loss focuses all gradient signal on producing better assistant responses, which is what evaluation measures.

**Why configurable separator tokens:** The tokens that delimit user and assistant turns differ by model family. ChatML (`<|im_start|>user\n` / `<|im_start|>assistant\n`) is used by Qwen, Llama-3-Instruct, and Mistral v0.3+. Llama-2 uses `[INST]`/`[/INST]`; Phi-3 uses `<|user|>\n`/`<|assistant|>\n`. The `instruction_part` and `response_part` keys in `sft.yaml` expose these as config rather than hardcoding them. See the comments in `configs/sft.yaml` for common presets.

**Why enable_thinking=False:** Qwen3.6-27B's chat template generates `<think>` tokens before answering by default. For SFT we want clean responses without thinking traces in the training labels (the Dolci-Instruct and catalog Q&A datasets don't contain thinking traces). `enable_thinking=False` is passed to `tokenizer.apply_chat_template()` with a `try/except TypeError` fallback — models whose tokenizer does not support this parameter (Llama, Mistral, Phi-3, etc.) silently ignore it.

**Why tool-use samples train tool calls:** SFT records that carry a top-level `tools` list (produced by `build_sft_qa.py --mode agentic`) are rendered with `apply_chat_template(..., tools=...)`, so the tool schemas and the assistant `tool_calls` land in the training text and the model actually learns to emit function calls. The loader reads and renders records in Python (rather than via `load_dataset`'s columnar inference), which both enables the `tools` path and sidesteps the arrow schema-mismatch that mixed plain-chat / tool-use rows would otherwise trigger.

**Why lower LR than CPT (2e-4 but still higher than CPT's 3e-5):** SFT is a smaller distributional shift than CPT — the model's weights are already adapted to Polish and we're now teaching response format and style. 2e-4 is standard for LoRA SFT. The reason it can be higher than CPT's 3e-5 is that CPT was starting from an instruct model whose alignment we wanted to preserve; at SFT time the model is already Polish-fluent and we want the instruction following to update more aggressively.

**Why r=32 (reduced from r=64):** SFT needs less adapter capacity than CPT. The heavy representational shift happened in CPT; SFT is a refinement. r=32 is sufficient for instruction-following adaptation and keeps the adapter smaller, which speeds up training and reduces the risk of overfitting on the 495k SFT examples.

**Why 3 epochs:** The SFT dataset (~500k examples) is much smaller than the CPT corpus. A single epoch would underfit — the model wouldn't generalize instruction-following patterns well. 3 epochs is the standard for LoRA SFT on datasets of this size; more than 3 epochs risks memorizing the template structure rather than generalizing.

```bash
# configs/sft.yaml already points base_model to models/cpt/merged
python scripts/train/sft.py --config configs/sft.yaml
# With the shipped config this runs 200 steps (a pilot), NOT the 3 epochs described above.
# Remove max_steps for the full run — see Full-run configuration.
# Checkpoints → models/sft/

# Merge:
python scripts/train/sft.py --config configs/sft.yaml --merge
# Output: models/sft/merged/
```

> To skip CPT and start directly from the base model (faster iteration),
> edit `configs/sft.yaml` and set `base_model:` to any HF model ID or local path.

---

## Step 8 — Preference Optimization (DPO)

**What it does:** Adjusts the model's response style and helpfulness using pairs of human-preferred and less-preferred responses for the same prompt. Operates on the SFT-merged model for one epoch at a very low learning rate.

**Why DPO (not PPO/RLHF):** Direct Preference Optimization eliminates the need for a separate reward model and the complex online RL training loop that PPO requires. In PPO, you train a reward model on preference data, then run rollouts from the policy, score them with the reward model, and compute policy gradients — four interacting components that each introduce failure modes. DPO reformulates the same objective as a direct supervised loss on the preference pairs. Given the size of our DPO dataset (225k pairs), PPO would be impractical; DPO runs in the same framework as SFT.

**Why very low LR (5e-6, 40× lower than SFT):** DPO is extremely sensitive to learning rate. A high LR causes reward hacking: the model learns to produce responses that score well on the preference metric but diverge significantly from the SFT policy, often degrading coherence or instruction following. The beta=0.1 parameter (KL divergence penalty) controls how far the DPO policy can drift from the SFT reference; a low LR keeps the updates small enough that beta has time to act as a brake.

**Why beta=0.1:** Beta is the KL penalty weight in the DPO loss. High beta (>0.5) makes the model barely update — the preference signal is overwhelmed by the penalty for diverging from SFT. Low beta (<0.05) allows the model to overfit to the preference pairs and collapse to a narrow output distribution. 0.1 is the standard from the original DPO paper and works well on datasets of this size.

**Why sigmoid loss (not IPO or KTO):** Sigmoid (standard DPO) is the most well-understood loss and has the most stable training behavior. IPO (Identity Preference Optimization) was proposed to fix theoretical issues with DPO's margin term but introduces different hyperparameter sensitivity. KTO (Kahneman-Tversky Optimization) can work with unpaired preferences, which is irrelevant here since Dolci-DPO provides matched pairs. Sigmoid DPO is the right default for standard paired preference data.

**Why 1 epoch:** DPO datasets are small relative to CPT/SFT corpora. More than 1 epoch on 225k pairs causes reward hacking even at low LR — the model starts optimizing for the training distribution of the Dolci-DPO pairs rather than generalizing to user preferences. One epoch is a well-established default for DPO.

```bash
python scripts/train/dpo.py --config configs/dpo.yaml
# 1 epoch — configs/dpo.yaml ships full-run sized (no max_steps). Checkpoints → models/dpo/

# Merge:
python scripts/train/dpo.py --config configs/dpo.yaml --merge
# Output: models/dpo/merged/
```

---

## Step 9 — Evaluate

**What it does:** Measures the fine-tuned model on four axes: Polish language tasks (fluency, reasoning, knowledge), English retention (catastrophic forgetting check), closed-book catalog knowledge (did the facts bake in?), and agentic tool-use (does it call the right tool with the right arguments?).

**Why separate evaluation suites:**

- **Polish suite** (`--suite polish`): Open PL LLM Leaderboard-style benchmarks (Belebele PL, ARC-PL, Global-MMLU PL, PIQA PL) measure whether the model actually improved at Polish — fluency, reasoning, and cultural knowledge. Without this, you can't distinguish "model is more Polish" from "model is just more verbose in Polish." `--suite polish_quick` runs the first two only.

- **English retention** (`--suite english`): Compares the fine-tuned model against the original Qwen/Qwen3.6-27B baseline on MMLU, HellaSwag, and ARC (English). The goal is that English scores don't drop more than 2–3 percentage points. If they drop more, the English replay fraction in CPT was insufficient.

- **Catalog knowledge (closed-book):** Tests whether facts from legal acts, GUS statistics, and dane.gov.pl descriptions actually entered the model's weights. The question set comes from the records carved out in [Step 1a](#1a-carve-out-the-evaluation-holdout--do-this-now) and deliberately excluded from training. This is the most direct test of the core project goal — knowledge injection without RAG. Scored with the numeric-aware `hybrid` scorer by default (or an LLM judge).

- **Agentic tool-use:** Tests whether the model selects the correct tool and emits well-formed, correct arguments (validated against the JSON Schema). `agentic_eval.py` parses the emitted `tool_calls` and reports `tool_selection_acc`, `args_exact_acc`, and `schema_valid_rate` — directly measuring the tool-use goal rather than token overlap.

```bash
# Polish tasks
python scripts/eval/run_eval.py \
    --model models/dpo/merged \
    --suite polish \
    --out eval/results

# English retention check — compare to the base model
python scripts/eval/run_eval.py \
    --model models/dpo/merged \
    --suite english \
    --out eval/results

python scripts/eval/run_eval.py \
    --model Qwen/Qwen3.6-27B \
    --suite english \
    --out eval/results/baseline

# vllm backend — higher throughput (merged model only, incompatible with --peft):
python scripts/eval/run_eval.py \
    --model models/dpo/merged \
    --suite polish \
    --backend vllm
```

Closed-book catalog knowledge, using the holdout carved out back in Step 1a:

```bash
# Build a held-out question set from the records kept out of training:
python scripts/process/build_sft_qa.py \
    --input "data/catalogs_holdout/**/*.jsonl" \
    --out eval/data/catalog_qa_holdout.jsonl \
    --mode template --per-record 1

# Score. Default scorer is `hybrid` — numeric-aware: when the reference contains numbers
# (GUS statistics) it requires them, instead of rewarding copied prose. Other scorers:
# overlap (token-overlap), numeric, or llm (judge model via --judge-model/--judge-base-url).
python scripts/eval/catalog_eval.py \
    --model models/dpo/merged \
    --qa eval/data/catalog_qa_holdout.jsonl \
    --scorer hybrid \
    --out eval/results/catalog

# vllm backend (batches all prompts at once — faster throughput):
python scripts/eval/catalog_eval.py \
    --model models/dpo/merged \
    --qa eval/data/catalog_qa_holdout.jsonl \
    --backend vllm

# GGUF backend (--model is the .gguf file path):
python scripts/eval/catalog_eval.py \
    --model models/gguf/model-Q4_K_M.gguf \
    --qa eval/data/catalog_qa_holdout.jsonl \
    --backend gguf
```

Agentic (tool-use) eval — does the model emit the *right* tool call?

```bash
python scripts/process/build_sft_qa.py \
    --input "data/catalogs_holdout/**/*.jsonl" \
    --out eval/data/agentic_holdout.jsonl \
    --mode agentic --per-record 1

python scripts/eval/agentic_eval.py \
    --model models/dpo/merged \
    --qa eval/data/agentic_holdout.jsonl \
    --out eval/results/agentic
# Reports: format_rate, tool_selection_acc, args_exact_acc, schema_valid_rate.
# --backend {hf,vllm,gguf} as above.
```

Both eval scripts support `--num-shards N --shard i` for a data-parallel fan-out across GPUs;
`scripts/eval/merge_shards.py --out <dir>` merges the results. `make eval-parallel
EVAL_GPUS=4` wires this up for you.

Results are written to `eval/results/`. Compare the fine-tuned model against
`Qwen/Qwen3.6-27B` (the untuned baseline) on all Polish tasks.

---

## Step 10 — Export to GGUF

**What it does:** Converts the **already-merged** DPO checkpoint to GGUF format and quantizes it to multiple bit-widths. The GGUF files are the final deliverable — self-contained model files that run in llama.cpp and Ollama without any Python dependencies.

**Prerequisite — run the DPO stage with `--merge` first.** Export reads a merged checkpoint (default `<config output_dir>/merged`, i.e. `models/dpo/merged`; override with `--model-dir`). It does not attach a fresh LoRA adapter to the config's `base_model` — an earlier bug that silently exported the *pre-fine-tune* weights. If the merged dir is missing, the script errors and tells you to run `--merge`.

**Why GGUF (not safetensors or ONNX):** GGUF is the universal format for llama.cpp and Ollama — the dominant serving stacks for self-hosted models. It embeds the tokenizer, chat template, and quantized weights in a single file, eliminating the need for a transformers install at inference time. ONNX is an alternative but requires ONNX Runtime, has weaker support for quantization at this scale, and doesn't natively support Qwen's SSM-hybrid architecture. Safetensors is the training format — keeping it means keeping the full bf16 model (~54 GB), which is impractical to distribute.

**Why multiple quantization levels:**

| Quant | Size | Use case |
|---|---|---|
| `Q4_K_M` | ~17 GB | Primary deployment — best quality/size tradeoff; fits in 24 GB VRAM |
| `Q5_K_M` | ~20 GB | Higher quality deployment when VRAM allows |
| `Q6_K` | ~23 GB | Near-lossless; for quality-sensitive applications |
| `Q8_0` | ~28 GB | Reference quality — used to verify quantization loss vs. f16 |

**Why two export backends:** Unsloth has its own GGUF exporter that runs within the same Python session as training, which is faster and avoids a separate llama.cpp compile. However, Unsloth may not support every architecture. The llama.cpp `convert_hf_to_gguf.py` approach always works regardless of architecture and is the safe fallback. `--backend auto` (the default) picks llamacpp unless Unsloth is in use, so it needs `--llama-dir` to point at a built llama.cpp checkout.

```bash
# Default: auto-select the backend (resolves to llamacpp for this project's configs)
python scripts/train/export_gguf.py \
    --config configs/dpo.yaml \
    --quants Q4_K_M Q5_K_M Q6_K Q8_0 \
    --out models/gguf \
    --llama-dir llama.cpp

# Force Unsloth's one-call exporter (if Qwen3.6 support is available)
python scripts/train/export_gguf.py \
    --config configs/dpo.yaml \
    --backend unsloth \
    --quants Q4_K_M Q5_K_M Q6_K Q8_0 \
    --out models/gguf
```

Smoke-test the output before considering it done:

```bash
python scripts/eval/smoke_gguf.py \
    --gguf models/gguf/model-Q4_K_M.gguf

# Or load directly in Ollama:
ollama create qwen-pl -f Modelfile   # write a Modelfile pointing to the GGUF
ollama run qwen-pl "Opisz krótko Konstytucję RP."
```

---

# Reference

## Understanding the core training knobs

Four config keys — `per_device_train_batch_size`, `gradient_accumulation_steps`,
`max_steps`, and `max_seq_len` — control the memory / speed / quality trade-off of every
training stage (CPT, SFT, DPO). They are the first things to touch when adapting the recipe
to a different GPU (e.g. a 48 GB card instead of the 96 GB reference rig).

**The number that actually matters — effective batch size:**

```
effective batch size = per_device_train_batch_size × gradient_accumulation_steps × num_GPUs
```

This is how many samples contribute to **one weight update**. The optimizer only sees this
product — but the two factors cost very different resources: **`per_device_train_batch_size`
costs VRAM, `gradient_accumulation_steps` costs wall-clock time.** The rule of thumb: raise
`per_device_train_batch_size` until VRAM is nearly full, then use
`gradient_accumulation_steps` to reach your target effective batch.

`num_GPUs` is the third factor, and it is the one you get for free: adding GPUs multiplies
the effective batch unless you compensate. By default the configs compensate for you —
`distributed.effective_batch: constant` divides `gradient_accumulation_steps` by the world
size so a 4-GPU run does exactly the same optimizer math as a 1-GPU run, four times faster.
See [Multi-GPU training](#multi-gpu-training).

| Knob | Scope | Larger → | Smaller → |
|---|---|---|---|
| **`per_device_train_batch_size`** | int ≥ 1 (VRAM-bound; ~1–2 for 27B @ seq 4096, more at seq 2048) | more VRAM used (activations scale ~linearly), better GPU utilization/throughput, less kernel-launch overhead — too high **OOMs** | less VRAM, OOM-safe, but underfeeds the GPU (worse utilization). `1` is the memory floor |
| **`gradient_accumulation_steps`** | int ≥ 1 (commonly 1–64) | bigger *effective* batch → smoother/more stable gradients, but **proportionally slower per logged step** (32 = 32 micro-batches before one update) and fewer updates per epoch | faster steps, more frequent updates, **noisier gradients**. `1` updates after every micro-batch |
| **`max_steps`** | int; `-1` / omit = run `num_train_epochs` fully | more of the dataset seen → more learning, longer run | finishes sooner. Only for smoke tests — 20 steps confirms the pipeline runs, it does **not** produce a usable model |
| **`max_seq_len`** | int, ~512–8192+ (≤ model context limit) | learns longer-range dependencies, but activation memory + attention compute grow with length (attention ~O(len²), SSM ~O(len)) → slower, more VRAM | cheaper/faster per token, less VRAM, but caps the context the weights adapt to |

**How they interact:**

- **VRAM (OOM levers):** driven by `per_device_train_batch_size × max_seq_len` (activations),
  on top of the fixed ~14 GB 4-bit weights + LoRA + optimizer states. Halving `max_seq_len`
  roughly halves activation memory, which can free room for a larger batch.
- **Wall-clock per step:** driven by `per_device_train_batch_size × gradient_accumulation_steps
  × max_seq_len` (total tokens processed before each update).
- **Model quality:** driven by the *effective batch size* (first two multiplied) and *how much
  data is seen* (`max_steps` / epochs × `max_seq_len`).

> One "step" = one **optimizer step** = `gradient_accumulation_steps` micro-batches. So
> `max_steps: 20` with `gradient_accumulation_steps: 4` runs 80 forward/backward passes.

**Practical recipe:**

- *Smoke test:* everything small (e.g. `per_device_train_batch_size: 1`,
  `gradient_accumulation_steps: 4`, `max_steps: 20`, `max_seq_len: 2048`) → steps in seconds,
  just enough to watch the loss fall and confirm the stack is healthy. This is what the repo
  ships.
- *Full run:* push `per_device_train_batch_size` as high as VRAM allows at your chosen
  `max_seq_len`, set `gradient_accumulation_steps` for an effective batch of ~16–32 (stable CPT),
  `max_seq_len: 4096`, and remove `max_steps` (let `num_train_epochs` govern length).

## Full-run configuration

The shipped configs are smoke-test sized. To do a real run, apply these edits — the intended
full-run values are already present as comments in each file.

**`configs/cpt.yaml`** — remove `max_steps: 20`, then:

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 16    # effective batch 32 on one GPU
max_seq_len: 4096                  # 96 GB allows ~8192 — raise after a pilot
# num_train_epochs: 1 then governs the run length
```

**`configs/sft.yaml`** — remove `max_steps: 200`, then:

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 8     # effective batch 32
# num_train_epochs: 3 then governs the run length
```

**`configs/dpo.yaml`** — already full-run sized (`bs=4 × ga=8`, `num_train_epochs: 1`, no
`max_steps`). No changes needed.

Then project the run length before you commit the GPUs:

```bash
python scripts/train/count_steps.py --config configs/cpt.yaml --num-gpus 4
```

See [Projecting run length](#projecting-run-length) for the arithmetic, and
[Checkpointing](#checkpointing--saving-resuming-and-stopping-safely) for how to survive a
multi-day run.

## Multi-GPU training

Append `NPROC=<n>` to any training target. `NPROC=1` (the default) is a plain single process:

```bash
make cpt NPROC=4          # 4-way DDP
make sft NPROC=4
make dpo NPROC=4
make cpt-merge            # merge is always single-process
```

Under the hood this is `torchrun --standalone --nproc_per_node=4`; `accelerate launch
--num_processes 4` works identically (both export `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, which is
all the scripts key off).

**Why DDP and not FSDP/DeepSpeed.** The 4-bit 27B is ~14 GB, so a full replica fits on one
96 GB card with room to spare — there is nothing to gain from sharding the model. That matters
here because **the RTX 6000 Pro Blackwell has no NVLink**: every byte between GPUs crosses
PCIe. DDP all-reduces only the LoRA gradients (tens of MB at r=64), which PCIe absorbs easily.
FSDP or ZeRO-3 would move whole layers each step and be badly bottlenecked.

**Effective batch.** Controlled by `distributed.effective_batch` in each config:

| Value | Behaviour |
|---|---|
| `constant` (default) | `gradient_accumulation_steps` is divided by the world size, so `per_device × grad_accum × num_GPUs` matches the single-GPU run exactly. Identical optimizer math, ~N× faster. No LR retuning. |
| `scale` | `gradient_accumulation_steps` is used as-is; the effective batch grows N×. Fewer optimizer steps over the same data — **retune `learning_rate` and `warmup_ratio`**. |

Project the step count for a given world size with
`python scripts/train/count_steps.py --config configs/cpt.yaml --num-gpus 4` — it applies the
same policy, so under `constant` the effective batch is unchanged and only the step count
drops.

**Other `distributed:` keys:** `ddp_find_unused_parameters` (leave `false`; set `true` only if
DDP reports unused parameters), `dataloader_num_workers` (per rank), `ddp_timeout` (raised to
5400 s because a cold 27B load can outlast the 1800 s NCCL default), and `device_map` (used
only for single-process runs, where `auto` still spreads one model across all visible cards).

**Prerequisites and gotchas:**

- Run `python scripts/check_env.py --min-gpus 4` first. It checks every card individually and
  reports NCCL availability.
- **NCCL is Linux-only.** Multi-GPU training must run on the Ubuntu box (or WSL2), not on a
  native Windows checkout.
- If NCCL hangs at initialization, inspect the topology with `nvidia-smi topo -m` and try
  `NCCL_P2P_DISABLE=1` — peer-to-peer over PCIe is the usual culprit on NVLink-less rigs.
- `--merge` refuses to run under a launcher: every rank would load the bf16 base and write the
  same `merged/` directory. Use the `*-merge` targets, which always use plain `python`.
- Only rank 0 prints. If you see four copies of every log line, the rank guard is not working.

**Eval** is data-parallel rather than tensor-parallel, for the same PCIe reason:

```bash
make eval-parallel EVAL_GPUS=4    # 4 single-GPU workers over disjoint slices, then merge
python scripts/eval/run_eval.py --model models/dpo/merged --suite polish --gpus 4
```

`--tp-size` is available on all three eval scripts but is only worth using when a model does
not fit on one card.

## Checkpointing — saving, resuming, and stopping safely

Checkpoints apply to every training stage (CPT / SFT / DPO) and are what make a multi-day
run survivable. Controlled by `save_steps` in each config.

**What gets saved:** every `save_steps` optimizer steps the trainer writes
`<output_dir>/checkpoint-<step>/` containing the **LoRA adapter weights** plus the
**optimizer, LR-scheduler, RNG, and trainer state** (`trainer_state.json`) — everything
needed to resume bit-exactly. Because only the adapter is saved (not the frozen 4-bit/bf16
base), each checkpoint is small (~hundreds of MB to a couple GB), not the full model. When
training completes, the final adapter is written to `<output_dir>/` itself; `--merge` then
folds it into `<output_dir>/merged/`.

**Why checkpoint frequently:** CPT is the long pole of the pipeline, and an SSH drop, OOM,
power blip, or NaN loss otherwise loses everything. `save_steps: 500` caps the worst-case loss
at ≤500 steps of work. Always launch long runs under **`tmux`/`nohup`** so a disconnected
shell doesn't kill the process, and treat the checkpoints as your rollback points.

**Resuming:** the checkpoint holds optimizer + scheduler + RNG state, so resuming continues
the *exact* LR curve and data order — not a fresh restart. The training scripts currently
call `trainer.train()` with no resume flag; to resume you must edit the call site:
```python
trainer.train(resume_from_checkpoint="models/cpt/checkpoint-2000")
```
Resume is only valid if the config is unchanged (batch, LR, schedule, data) — the scheduler
state assumes the same total-step horizon it was created with. See
[Known limitations](#known-limitations).

**Disk — checkpoints are pruned to `save_total_limit`:** the configs set `save_total_limit: 3`,
so only the most recent 3 `checkpoint-*` dirs are kept and older ones are deleted automatically
(raise it if you want more rollback points, at the cost of disk).

**Automatic early-abort on instability (`stability_guard`):** a too-high LR (or a bad batch)
shows up as exploding `grad_norm` and rising/NaN loss — left alone, a multi-day run keeps
burning hours going nowhere. All three trainers attach a `StabilityGuard` callback
(`scripts/train/_common.py`) that watches the per-step training log and **stops the run** on
NaN loss or after `stability_patience` consecutive `grad_norm` spikes above
`max_grad_norm_abort`. It fires at `logging_steps` cadence, so a diverging run halts in a
handful of steps (~seconds–minutes) instead of hours, and your `save_total_limit` checkpoints
remain the rollback points — resume from the last good one after lowering the LR. Config keys
(each overridable by an env var, env wins — steer a run without editing YAML):

| Config key | Default | Env override |
|---|---|---|
| `stability_guard` | `true` | `PSLAB_STABILITY_GUARD=0` / `1` |
| `max_grad_norm_abort` | `100.0` | `PSLAB_MAX_GRAD_NORM_ABORT=<float>` |
| `stability_patience` | `3` | `PSLAB_STABILITY_PATIENCE=<int>` |

```bash
# e.g. run a jittery config on a tighter leash without touching the YAML:
PSLAB_MAX_GRAD_NORM_ABORT=50 PSLAB_STABILITY_PATIENCE=2 python scripts/train/cpt.py --config configs/cpt.yaml
```
(This is distinct from TRL's `max_grad_norm` gradient *clipping*, which bounds each step but never stops the run. The guard is a *circuit breaker*; note the guard reads the **pre-clip** `grad_norm` the trainer logs.)

**Stopping early on purpose — set `max_steps`, don't Ctrl-C mid-cosine:** the cosine scheduler anneals
the LR toward ~0 over its full horizon, and that final low-LR annealing is where a lot of
the settling happens. If you decide up front to run fewer steps, set `max_steps: N` in the
config so the anneal *completes* over N steps (a clean checkpoint). Killing a longer run
partway leaves the LR only partly decayed — a mid-schedule, under-annealed checkpoint that
is measurably worse than a run properly scheduled for N. (`max_steps` overrides
`num_train_epochs`; note one epoch of the CPT mix is a fixed step count the trainer prints
as `Total optimization steps` at startup.)

**Evaluate a checkpoint without stopping the run:** because each checkpoint is a
self-contained adapter, you can point the evaluator at one mid-run while training continues —
`run_eval.py --peft models/cpt/checkpoint-2000 --base-model <base>` — to test an intermediate
model and decide whether the remaining steps are worth it.

---

# Development

## Makefile — one entrypoint per stage

The `Makefile` is a **task runner** (not a build system — there's nothing to compile). Each
target wraps the canonical command for a pipeline stage so runs are reproducible from one
place instead of copy-pasting the long `python scripts/...` lines. Run it from the repo root:

```bash
make help          # list all targets
make env           # → python scripts/check_env.py
make cpt           # → python scripts/train/cpt.py --config configs/cpt.yaml
make cpt-merge     # → ...cpt.py --config configs/cpt.yaml --merge
```

**Override the defaults** with `VAR=value` on the command line (`?=` vars: `PY`, `CPT_CFG`,
`SFT_CFG`, `DPO_CFG`, `NPROC`, `EVAL_GPUS`):

```bash
make cpt CPT_CFG=configs/cpt_gemma.yaml   # different config
make cpt PY=python3.12                     # different interpreter
make cpt NPROC=4                           # 4-way DDP (see Multi-GPU training)
make eval-parallel EVAL_GPUS=4             # one eval worker per card
```

**Targets, grouped** (intended order, top-to-bottom):

| Group | Targets |
|---|---|
| dev | `env` · `test` · `lint` |
| data | `ingest` · `process` · `dedup` · `build-cpt` · `build-sft` · `build-agentic` |
| training | `cpt` / `cpt-merge` · `sft` / `sft-merge` · `dpo` / `dpo-merge` (add `NPROC=n`) |
| eval + export | `eval` · `eval-parallel` · `eval-agentic` · `gguf` |

> **⚠ The data targets are simplified and lag the commands in Steps 1–4.** In particular
> `build-cpt` uses the bare `*.jsonl` glob (misses the gzipped `*.jsonl.gz` dedup shards →
> drops the whole web corpus), omits `--max-per-source`, and — like `build-sft` and
> `build-agentic` — reads `data/catalogs/**` rather than the `data/catalogs_train/**` split,
> **which leaks the eval holdout**. `ingest` uses minimal args. There are also no targets for
> `make_holdout.py`, `build_dpo.py`, `count_steps.py`, `run_eval.py`, or `smoke_gguf.py`.
>
> **Use `make` for the training / eval / export stages** (they read the configs, so they honor
> every config edit), but **build the datasets with the explicit Step 1–4 commands** above
> until the Makefile targets are brought in line.

## Lint and tests

Pure-logic modules (license filtering, tool schemas + validation, LoRA target detection,
tool-call parsing, dedup band math, Polish quality heuristics, distributed arg policy, chat
rendering) have unit tests that need no GPU or training stack:

```bash
pip install ruff pytest jsonschema pyyaml requests   # light deps only
make lint      # ruff check scripts tests
make test      # pytest
```

`ruff`/`black`/`pytest` are configured in `pyproject.toml`. `.github/workflows/ci.yml` runs
lint + tests, but **only on manual dispatch** — the `push` / `pull_request` triggers are
currently commented out. Run it with `gh workflow run ci.yml` or from the Actions tab, and
re-enable the automatic triggers by uncommenting them in that file.

---

# Quick reference — all commands in order

This is a faithful condensation of the walkthrough above, not a variant recipe.

```bash
# 0. env  (installing into the existing vllm venv — torch/vllm/flashinfer already present)
pip install -r requirements.txt
python scripts/check_env.py

# 1. catalogs
python scripts/ingest/sejm_isap.py --publisher DU --years 2015-2024 --out data/catalogs/isap/du_2015_2024.jsonl
python scripts/ingest/sejm_isap.py --publisher MP --years 2015-2024 --out data/catalogs/isap/mp_2015_2024.jsonl
python scripts/ingest/dane_gov.py  --out data/catalogs/dane_gov/datasets.jsonl --max-pages 200 --commercial-safe
python scripts/ingest/gus_bdl.py   --list-subjects   # discover valid IDs first
python scripts/ingest/gus_bdl.py   --subjects K11,K15,K27,K43,K47,K44,K23,K24,K54,K3,K9,K20,K21,K8,K10,K22 \
    --years 2010-2025 --max-vars-per-subject 300 --delay 0.6 --out data/catalogs/gus_bdl/indicators.jsonl

# 1a. carve out the eval holdout NOW — everything downstream reads data/catalogs_train/
python scripts/process/make_holdout.py --input "data/catalogs/**/*.jsonl" \
    --train-out data/catalogs_train --holdout-out data/catalogs_holdout --fraction 0.02

# 2. corpora  (HF download — see Step 2 above)

# 3. process
python scripts/process/pipeline.py --input "data/raw/**/*.jsonl" --output data/interim/clean --workers 16 --lang-threshold 0.7
python scripts/process/dedup.py    --input data/interim/clean --output data/interim/dedup --workers 16 --threshold 0.8

# 4. datasets   (*.jsonl* — dedup output is gzipped; cap the dominant GUS source)
python scripts/process/build_cpt_mix.py --pl "data/interim/dedup/**/*.jsonl*" "data/catalogs_train/**/*.jsonl" \
    --en "data/raw/replay_en/**/*.jsonl" --out data/processed/cpt \
    --replay-fraction 0.18 --max-per-source gus_bdl=1000000 --commercial-safe
python scripts/process/build_sft_qa.py  --input "data/catalogs_train/**/*.jsonl" --out data/processed/sft/catalog_qa.jsonl --mode template --per-record 2
python scripts/process/build_sft_qa.py  --input "data/catalogs_train/**/*.jsonl" --out data/processed/sft/agentic/tool_qa.jsonl --mode agentic --per-record 2
python scripts/process/build_dpo.py     --input data/raw/dolci-dpo-pl/data.jsonl --out data/processed/dpo --val 500
cat data/raw/dolci-sft-pl/data.jsonl data/processed/sft/catalog_qa.jsonl data/processed/sft/agentic/tool_qa.jsonl > data/processed/sft/train.jsonl
head -1000 data/processed/sft/train.jsonl > data/processed/sft/val.jsonl

# 5. pilot (configs ship with max_steps: 20 — confirm the stack works)
python scripts/train/cpt.py --config configs/cpt.yaml

# 6-8. train  (apply Full-run configuration first, or these stay smoke tests)
python scripts/train/cpt.py --config configs/cpt.yaml && python scripts/train/cpt.py --config configs/cpt.yaml --merge
python scripts/train/sft.py --config configs/sft.yaml && python scripts/train/sft.py --config configs/sft.yaml --merge
python scripts/train/dpo.py --config configs/dpo.yaml && python scripts/train/dpo.py --config configs/dpo.yaml --merge

# 9. eval  (--backend vllm for faster throughput; --backend gguf for GGUF models)
python scripts/eval/run_eval.py --model models/dpo/merged --suite polish
python scripts/eval/run_eval.py --model models/dpo/merged --suite english
python scripts/process/build_sft_qa.py  --input "data/catalogs_holdout/**/*.jsonl" --out eval/data/catalog_qa_holdout.jsonl --mode template --per-record 1
python scripts/process/build_sft_qa.py  --input "data/catalogs_holdout/**/*.jsonl" --out eval/data/agentic_holdout.jsonl --mode agentic --per-record 1
python scripts/eval/catalog_eval.py --model models/dpo/merged --qa eval/data/catalog_qa_holdout.jsonl --scorer hybrid
python scripts/eval/agentic_eval.py --model models/dpo/merged --qa eval/data/agentic_holdout.jsonl

# 10. export
python scripts/train/export_gguf.py --config configs/dpo.yaml --quants Q4_K_M Q5_K_M Q6_K Q8_0 --out models/gguf --llama-dir llama.cpp
python scripts/eval/smoke_gguf.py   --gguf models/gguf/model-Q4_K_M.gguf
```

---

# Project layout

```
configs/          cpt.yaml  sft.yaml  dpo.yaml
  models/         qwen3_ssm.yaml  llama3.yaml  phi3.yaml  (LoRA presets)
scripts/
  check_env.py
  common/         records.py  tool_catalog.py  tooling.py   (schema, tools, validation)
  ingest/         sejm_isap.py  dane_gov.py  gus_bdl.py  culturax_pl.py  hplt_pl.py  replay_en.py
  process/        pipeline.py  dedup.py  build_cpt_mix.py  build_sft_qa.py  build_dpo.py
                  make_holdout.py  quality_pl.py
  train/          cpt.py  sft.py  dpo.py  export_gguf.py  count_steps.py  _common.py
  eval/           run_eval.py  catalog_eval.py  agentic_eval.py  merge_shards.py  smoke_gguf.py
tests/            unit tests for the pure-logic modules (pytest)
data/
  raw/            downloaded corpora (gitignored)
  interim/        clean/  dedup/  (gitignored)
  processed/      cpt/  sft/  dpo/  (gitignored)
  catalogs/       ISAP  dane.gov.pl  GUS BDL — unsplit ingest output  (gitignored)
  catalogs_train/ training split from make_holdout.py — what the builders read  (gitignored)
  catalogs_holdout/  eval holdout, deliberately NOT under catalogs/  (gitignored)
models/           adapters  merged  gguf  (gitignored)
eval/results/     benchmark scores  (gitignored)
docs/             SETUP.md          (environment setup — start here)
plans/            design/implementation notes  (gitignored)
Makefile          per-stage entrypoints        pyproject.toml   ruff/black/pytest config
.github/workflows/ci.yml   lint + tests (manual dispatch only)
LICENSE           Apache-2.0
```

---

# Troubleshooting

| Symptom | Fix |
|---|---|
| `check_env.py` bf16 test fails | PyTorch lacks `sm_120` kernels — reinstall from the `cu128` index (matches the pinned `nvidia-*-cu12==12.8.x` wheels) |
| `Unsloth unavailable` in training log | Unsloth may not support your architecture; PEFT fallback is used automatically — no action needed |
| OOM during CPT | Reduce `per_device_train_batch_size` to 1, double `gradient_accumulation_steps` |
| Multi-GPU run hangs before step 1 | Peer-to-peer over PCIe — check `nvidia-smi topo -m`, retry with `NCCL_P2P_DISABLE=1` |
| `target_modules: auto detected no known projection modules` | Auto-detection found nothing and raised (listing candidate module names) — set `lora.target_modules` explicitly from a `configs/models/` preset |
| Training finishes suspiciously fast | `max_steps` is still set — the configs ship pilot-sized. See [Full-run configuration](#full-run-configuration) |
| Catalog eval scores implausibly high | You trained on the holdout. Rebuild the datasets from `data/catalogs_train/**`, not `data/catalogs/**` — see [Step 1a](#1a-carve-out-the-evaluation-holdout--do-this-now) |
| GGUF smoke test produces garbled text | Chat template mismatch — ensure `<\|im_start\|>` tokens are in the GGUF's tokenizer |
| GGUF export behaves like the base model | You didn't merge — run the final stage with `--merge` first; export reads `<output_dir>/merged` (or pass `--model-dir`) |
| lm-eval task not found | Run `lm-eval --tasks list \| grep -i pl` to get current Polish task IDs |
| English scores drop >3% after CPT | Rebuild the mix with `--replay-fraction 0.22` on `build_cpt_mix.py`, then restart CPT from the last clean checkpoint |
| DPO loss spikes or diverges | LR is too high — halve `learning_rate` in `configs/dpo.yaml` and restart from SFT merged |
| `--backend vllm` with `--peft` | Not supported — vllm cannot load PEFT adapters; use a merged model with `--model` |

---

# Known limitations

Things that are true of the current code and worth knowing before you rely on them.

- **No resume flag.** All three trainers call `trainer.train()` with no arguments. To resume
  from a checkpoint you must edit the call site in `scripts/train/{cpt,sft,dpo}.py` — see
  [Checkpointing](#checkpointing--saving-resuming-and-stopping-safely). A `--resume` CLI flag
  does not exist yet.
- **In-trainer English retention check is not implemented.** `english_retention_check` appears
  (commented out) in `configs/cpt.yaml` but nothing in `scripts/train/` reads it. Retention
  must be measured out-of-band with `run_eval.py --suite english` against a checkpoint.
- **Makefile data targets lag the walkthrough** and one of them leaks the eval holdout — see
  the warning under [Makefile](#makefile--one-entrypoint-per-stage). Build datasets with the
  explicit commands.
- **CI does not run automatically.** `push` / `pull_request` triggers are commented out in
  `.github/workflows/ci.yml`; it is manual-dispatch only.
- **lm-eval task IDs are unverified** against the pinned `lm_eval==0.4.12`. `run_eval.py` says
  so in a source comment. Confirm with `lm-eval --tasks list | grep -i pl` before trusting a
  suite to run.
- **The configs ship pilot-sized** (`cpt.yaml: max_steps: 20`, `sft.yaml: max_steps: 200`).
  See [Full-run configuration](#full-run-configuration).
- **Unsloth support for Qwen3.6 (`Qwen3_5` arch) may not exist yet**, so `use_unsloth: false`
  is the default and the PEFT + bitsandbytes path is the primary one.

---

# License

**Code:** [Apache-2.0](LICENSE).

**Data and model weights are a separate question.** The pipeline downloads corpora under
several different licenses, and those obligations flow through to anything you train:

| Source | License | Commercial use | Notes |
|---|---|---|---|
| Sejm/ISAP legal acts | public domain | yes | Polish legal acts are not copyrightable |
| GUS BDL statistics | public domain | yes | |
| dane.gov.pl descriptions | CC-BY / CC0 | yes | attribution for CC-BY |
| HPLT 2.0 Polish | CC0 | yes | |
| C4 (English replay) | CC-BY | yes | attribution |
| Polish Wikipedia | **CC-BY-SA** | yes, with **share-alike** | copyleft — see caveat below |
| CulturaX Polish | **ODC-BY** | yes, with attribution | database-rights terms |
| Dolci-Instruct SFT / DPO | Apache-2.0 | yes | |

**`--commercial-safe` is a filter, not a clearance.** It is available on `dane_gov.py` and
`build_cpt_mix.py` and works off the `license` field that every record carries (see
`scripts/common/records.py`). It rejects NC (non-commercial) and ND (no-derivatives) terms —
**and also rejects records whose license is unknown**, which is the conservative default. It
does **not** strip CC-BY-SA or ODC-BY content, because those *are* commercially usable. So a
`--commercial-safe` CPT mix can still contain share-alike (Wikipedia) and database-attribution
(CulturaX) material, and the resulting merged weights are not unencumbered. If you need clean
provenance, exclude those corpora at mix time and keep the per-record `source` / `license` /
`snapshot_date` metadata as your audit trail.

**Base model:** Qwen3.6-27B carries its own license from Alibaba, which governs the derived
weights and GGUF artifacts regardless of anything above. Check it before redistributing.

Nothing here is legal advice.
