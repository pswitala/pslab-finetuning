# Qwen3.6-27B → Polish

Fine-tune **Qwen3.6-27B** for the Polish context: native fluency, cultural
grounding, and factual knowledge from Polish open-data catalogs baked into the
weights. Final artifact: **GGUF** files runnable in llama.cpp / Ollama.

**Hardware:** 1× NVIDIA RTX 6000 Pro Blackwell, 96 GB VRAM (single GPU, QLoRA).

**Architecture note:** Qwen3.6-27B is an SSM-hybrid VLM (`Qwen3_5ForConditionalGeneration`):
64 layers = 16 × (3× Gated DeltaNet + 1× Gated Attention). Only an instruct
checkpoint exists — no base. Vision encoder is frozen automatically during training.

---

## Step-by-step guide

### Step 0 — Install the environment

**What it does:** Prepares the Python environment with training-specific packages on top of the existing `vllm` virtualenv, which already ships with a Blackwell-compatible torch and inference stack.

**Why this approach:** The vllm venv already has `torch==2.10.0` built against CUDA 12.9 with `sm_120` (Blackwell) kernels and `flashinfer-python==0.6.6` as the attention backend. Starting from scratch would mean rebuilding or downloading a large CUDA-enabled torch wheel unnecessarily. We only add what the training pipeline needs: `peft`, `trl`, `accelerate`, `bitsandbytes`, and the corpus tooling.

**Why not flash-attn:** `flashinfer` is the Blackwell-compatible replacement for Flash Attention. The `flash-attn` package does not support `sm_120` as of planning time and conflicts with flashinfer if both are installed.

**Why bitsandbytes for quantization:** bitsandbytes is the standard 4-bit NF4 training backend. Alternatives like GPTQ and AWQ produce inference-only quantized weights — you cannot attach trainable LoRA adapters on top of them. Only bitsandbytes keeps the quantized base frozen while training full-precision LoRA adapters alongside it.

```bash
# 1. Activate the vllm venv (adjust path to match your installation)
source /path/to/vllm-venv/bin/activate

# 2. torch is already installed — skip reinstalling it.
#    Verify it sees the GPU:
python3 -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability())"
# expect: 12.9  (12, 0)

# 3. Install training packages that are missing from the vllm venv
pip install -r requirements.txt
# installs: peft trl accelerate bitsandbytes datasets pyarrow zstandard datatrove trafilatura lm-eval

# 4. (Optional) Install llama.cpp for GGUF export fallback
#    Only needed if Unsloth's built-in GGUF export is unavailable.
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j$(nproc)

# 5. Make scripts executable
chmod +x scripts/check_env.py scripts/**/*.py

# 6. Authenticate with Hugging Face (to pull Qwen3.6-27B)
huggingface-cli login

# 7. Copy and fill in .env
cp .env.example .env
# Edit .env: set HF_TOKEN, optionally WANDB_API_KEY
```

> **Note on flash-attn:** do **not** install `flash-attn` — `flashinfer-python==0.6.6`
> is already present and is the correct Blackwell-compatible replacement. Installing
> both will cause conflicts.

Verify the GPU and stack before doing anything else:

```bash
python scripts/check_env.py
# Expected: RTX 6000 Pro Blackwell, sm_120, ~96 GB, bf16 matmul OK
```

---

### Step 1 — Download open-data catalogs

**What it does:** Harvests Polish factual text from three government sources and stores each record with provenance metadata (`source`, `license`, `snapshot_date`). This text becomes two inputs: raw training signal for CPT and the ground truth for synthetic Q&A in SFT.

**Why catalog knowledge is injected via training (not RAG):** The goal is a model that answers questions about Polish law and statistics without needing any retrieval system at inference time. RAG is simpler to add but creates a deployment dependency and fails silently when the retrieval index is stale or wrong. Baking facts into weights is harder but produces a fully self-contained model, which is a requirement for the GGUF/Ollama deployment target.

**Why these three sources:**
- **Sejm/ISAP** — the official Polish legal journal. Acts from Dziennik Ustaw (DU) and Monitor Polski (MP) are public-domain, authoritative, and written in formal Polish — high-signal training text.
- **dane.gov.pl** — the national open-data portal. Dataset descriptions are concise Polish prose covering diverse domains (health, agriculture, environment) and carry explicit CC-BY / CC0 licenses, so they are safe for commercial use.
- **GUS BDL** — Statistics Poland's statistical database. ~5.5 M indicator-value pairs covering demographics, economy, and infrastructure from 2010 onward. Public-domain and verbalized as Polish sentences — dense factual coverage that CPT and catalog Q&A both benefit from.

**Why provenance per record:** The `license` field on every record lets `build_cpt_mix.py` run `--commercial-safe` filtering at mix time without re-downloading data. The `snapshot_date` defines the model's knowledge cutoff precisely, which is important to communicate to end users.

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

---

### Step 2 — Download pre-training corpora

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

### Step 3 — Clean and deduplicate the corpus

**What it does:** Removes low-quality documents, non-Polish text, and near-duplicate content across all corpora before any training data is assembled.

#### 3a — Quality filtering (`pipeline.py`)

**What it does:** Passes every document through four sequential filters: language detection, repetition detection, general quality heuristics, and Polish-specific heuristics.

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

#### 3b — Cross-corpus MinHash deduplication (`dedup.py`)

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
    --workers 16
```

---

### Step 4 — Build training datasets

#### 4a. CPT mixture (`build_cpt_mix.py`)

**What it does:** Combines cleaned Polish documents with a controlled fraction of English text and writes the result as sharded parquet files ready for the trainer.

**Why parquet (not JSONL):** The training loop loads data via HuggingFace `datasets`, which reads parquet efficiently via Apache Arrow's columnar format. Random access into a 400 GB JSONL file is slow; parquet partitioned into 100k-row shards can be memory-mapped and accessed at the row level. This matters because the trainer needs to shuffle across the entire dataset during each epoch.

**Why 18% English replay:** The replay fraction was set to balance two competing risks. Too little English (<10%) and the model forgets English reasoning chains and MMLU-style tasks after one CPT epoch — measurable via the English retention check that runs every 500 steps. Too much English (>25%) wastes training compute on a language the model already knows well, reducing the effective Polish exposure per GPU-hour. The 15–20% range is consistent with what multilingual CPT papers (GlotLID, EMMA-500) report as effective for adding a new language while preserving existing ones.

**Why no tokenization at this stage:** The trainer uses `packing=True` during CPT, which concatenates documents with a separator token and packs them into full-length sequences. This is more efficient than pre-tokenizing: you avoid padding waste and the trainer can rebalance sequence lengths across batches. Pre-tokenizing would also couple the dataset to a specific `max_seq_len`, making it harder to experiment with different context lengths.

**Why shuffle before writing:** The Polish and English documents are interleaved at the document level (not the batch level) so each training batch sees a mixture. If they were written as separate blocks, the model would oscillate between pure-Polish and pure-English training signals, which is worse for the replay mechanism.

```bash
python scripts/process/build_cpt_mix.py \
    --pl "data/interim/dedup/**/*.jsonl" \
         "data/catalogs/**/*.jsonl" \
    --en "data/raw/replay_en/**/*.jsonl" \
    --out data/processed/cpt \
    --replay-fraction 0.18 \
    --commercial-safe
# Output: data/processed/cpt/train/*.parquet  +  .../val/*.parquet
```

#### 4b. SFT instruction dataset (`build_sft_qa.py`)

**What it does:** Generates synthetic Polish Q&A pairs from catalog records and merges them with the downloaded Dolci-Instruct dataset. Each catalog record produces up to 2 question-answer pairs using source-specific templates (different templates for legal acts vs. GUS statistics vs. dane.gov.pl).

**Why template-based QA (not LLM-generated):** Three reasons. First, license safety — questions generated by a commercial LLM inherit that LLM's terms of service, which may restrict use. Template-generated questions are entirely original. Second, controllability — templates ensure every Q&A pair is strictly grounded in the source record; LLM-generated answers sometimes add facts not in the source. Third, no dependency — running a teacher LLM for 5.5M records would cost significant money and add infrastructure complexity. The template approach is deterministic, free, and reproducible.

**Why per-record 2:** One question per record is a wasted opportunity given the diversity of question templates. More than 2 risks the SFT dataset becoming dominated by formulaic catalog Q&A, which could hurt general instruction following. Two pairs per record gives coverage diversity while keeping the catalog fraction of the SFT mix reasonable relative to the 495k Dolci-Instruct examples.

**Why cat rather than a merge script:** The Dolci-Instruct data is already in chat format; the catalog Q&A output is also chat format. Shell `cat` is sufficient for the merge, and the val split is taken as the first N lines before the merge would contaminate val with training data. If the format ever changes, replace the cat with a Python merge script.

```bash
# Generate synthetic Q&A from catalog records
python scripts/process/build_sft_qa.py \
    --input "data/catalogs/**/*.jsonl" \
    --out data/processed/sft/catalog_qa.jsonl \
    --mode template --per-record 2

# Merge with downloaded instruction datasets
cat data/raw/dolci-sft-pl/data.jsonl \
    data/processed/sft/catalog_qa.jsonl \
    > data/processed/sft/train.jsonl
# Create a small val split (e.g. first 1000 lines)
head -1000 data/processed/sft/train.jsonl > data/processed/sft/val.jsonl
```

#### 4c. DPO preference dataset

```bash
cp data/raw/dolci-dpo-pl/data.jsonl data/processed/dpo/train.jsonl
# val split
head -500 data/processed/dpo/train.jsonl > data/processed/dpo/val.jsonl
```

---

### Step 5 — Pilot run (strongly recommended before the full run)

**What it does:** Runs 200 training steps through the full CPT pipeline — model loading, 4-bit quantization, LoRA attachment, forward pass, gradient computation, checkpoint save — and exits cleanly.

**Why run a pilot before committing to multi-day training:** Blackwell's `sm_120` compute capability is new enough that some CUDA kernel paths have subtle incompatibilities. A 200-step pilot catches: OOM from incorrect batch size estimates, PEFT `target_modules` errors if layer names differ from what the config expects, bitsandbytes CUDA errors from version mismatches, and gradient overflow with bf16. Finding any of these 200 steps in takes a few minutes; finding them 12 hours into a CPT run means restarting from scratch.

**Why 200 steps specifically:** Enough to see loss decrease monotonically (confirming gradients flow correctly) and trigger at least one checkpoint save (confirming disk writes work), but not so many that a failed pilot wastes GPU time.

```bash
# Quick pilot: 200M tokens CPT, 5k SFT examples
# Temporarily override in configs/cpt.yaml:
#   num_train_epochs: 1  →  max_steps: 200
#   per_device_train_batch_size: 1

python scripts/train/cpt.py --config configs/cpt.yaml
# Watch: loss should decrease, no CUDA errors, no OOM
```

If you hit OOM, reduce `per_device_train_batch_size` and increase
`gradient_accumulation_steps` to compensate (keep effective batch size the same).

**CPT pilot — training metrics**

Observed on the 200-step pilot (RTX 6000 Pro Blackwell, QLoRA r=64, packing, bs=2 × ga=8):
33.24 s/step, 3.6 h training + 1.7 h eval = **5.3 h total**. At this step time, the full
~840 k-step CPT run is approximately 7.7 days of training (eval overhead depends on
`eval_steps` and does not scale linearly).

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

**Pilot eval — CPT results**

After the 200-step CPT pilot, run the quick Polish eval to confirm nothing is broken:

```bash
# With PEFT adapter (fast — base loads from HF cache):
python scripts/eval/run_eval.py --peft models/cpt --suite polish_quick \
    --base-model /home/pswitala/models/cpt-merged

# Or after merging (if model lives on Linux fs):
python scripts/eval/run_eval.py --model /home/pswitala/models/cpt-merged --suite polish_quick
```

Observed results from the CPT pilot run on this machine:

| Task | Metric | Baseline (`Qwen/Qwen3.6-27B`) | CPT pilot (200 steps) | Change |
|---|---|---|---|---|
| `belebele_pol_Latn` | acc | 93.0% | 92.5% | −0.5% |
| `arc_challenge_mt_pl` | acc_norm | 52.0% | 50.5% | −1.5% |

**How to interpret these numbers:**

200 steps is less than 0.03% of the full ~840k-step CPT run. Benchmark improvements from CPT only become visible after the model has seen enough Polish text to shift its representations — typically after several thousand steps. A pilot showing no change or a small dip is normal and expected. The pilot's job is to confirm the training stack is healthy, not to show improved Polish scores.

| Metric change (pilot scale) | Meaning | Action |
|---|---|---|
| ±2% on either benchmark | Noise — expected | Continue to full CPT |
| −3% to −5% on `acc_norm` | Early degradation signal | Monitor; may need higher replay fraction |
| > −5% drop, or NaN loss | Something is wrong | Stop — debug LR, replay data, or bf16 overflow |

---

### Step 6 — Continued Pretraining (CPT)

**What it does:** Trains the model on raw Polish text (web corpora, Wikipedia, legal acts, GUS statistics) plus English replay using next-token prediction on packed sequences. Produces a Polish-fluent model that still follows instructions. This is the most compute-intensive stage.

**Why start from the instruct checkpoint (not a base model):** Qwen3.6-27B has no publicly released base checkpoint — only the instruct model exists. This is unusual but works, with adjustments. Starting from instruct means the model already knows how to follow instructions; CPT risks eroding this if done carelessly.

**Why conservative learning rate (3e-5 vs. typical 1e-4):** A high LR in CPT on an instruct model causes alignment degradation — the model stops following instructions and generates coherent Polish prose but ignores user intent. 3e-5 gives enough gradient signal to adapt to Polish text distribution while keeping the instruction-following behavior mostly intact. The English replay also helps anchor the model to its original capabilities.

**Why QLoRA (not full fine-tuning):** At 27B parameters and bf16 precision, the model alone occupies ~54 GB of VRAM. Full fine-tuning requires optimizer states (Adam: 2× model size) on top — well over 96 GB. QLoRA loads the base model in 4-bit NF4 (~14 GB), keeps frozen, and attaches trainable 16-bit LoRA adapters (~1-2 GB depending on rank). This fits comfortably within 96 GB while still adapting 27B parameters of representation capacity through the adapter layers.

**Why LoRA must target both attention types (SSM and standard):** Qwen3.6-27B's 64 layers are 75% Gated DeltaNet SSM layers (`linear_attn.*`) and only 25% standard attention layers (`self_attn.*`). Targeting only `q_proj`/`k_proj`/`v_proj` (the standard attention approach from Llama-era LoRA recipes) leaves 48 of 64 layers completely frozen — effectively a 75% capacity reduction. The config targets both `self_attn.*` projections and `linear_attn.*` projections (`in_proj_qkv`, `out_proj`, `in_proj_z`, `in_proj_a`, `in_proj_b`).

**Why rsLoRA (`use_rslora: true`):** Standard LoRA scales the adapter output by `alpha/r`. At high rank (r=64), this scaling can cause gradient instability during the early warmup steps. rsLoRA replaces this with `alpha/sqrt(r)`, which remains stable across a wider range of ranks and makes it safe to use r=64 for CPT without careful per-run LR tuning.

**Why r=64 for CPT (higher than SFT's r=32):** CPT is adapting the model to a new language distribution — a larger distributional shift than instruction following. Higher rank gives more expressiveness in the adapter. The compute cost is acceptable at CPT's 1-epoch schedule; for SFT's 3-epoch schedule, the smaller r=32 is sufficient.

**Why packing=True:** CPT operates on raw text documents of varying length. Without packing, each training sequence would be padded to `max_seq_len`, wasting 30–60% of compute on padding tokens. Packing concatenates documents with a separator and fills sequences to capacity, achieving near-100% GPU utilization. This is only appropriate for CPT (where document boundaries don't affect the learning objective). SFT must not use packing because loss masking on responses would span document boundaries incorrectly.

**Why English retention check every 500 steps:** CPT on a monolingual Polish corpus on top of a multilingual instruct model risks catastrophic forgetting. Running a small English MMLU/HellaSwag/ARC evaluation every 500 steps creates a real-time signal: if English accuracy drops more than ~3% below baseline, the replay fraction should be increased and training restarted from the last clean checkpoint.

```bash
python scripts/train/cpt.py --config configs/cpt.yaml
# Runs for ~1 epoch over the full CPT mixture.
# Checkpoints saved to models/cpt/ every 500 steps.
# English retention benchmark runs every 500 steps (watch for forgetting).

# When training finishes, merge the adapter into full weights:
python scripts/train/cpt.py --config configs/cpt.yaml --merge
# Output: models/cpt/merged/
```

---

### Step 7 — Supervised Fine-Tuning (SFT)

**What it does:** Teaches the model to follow Polish instructions and answer questions about catalog knowledge in a conversational format. Starts from the CPT-merged weights, trains on chat-formatted examples, and computes loss only on assistant responses.

**Why start from CPT merged (not the original instruct model again):** CPT gave the model Polish fluency and catalog knowledge. Starting SFT from CPT/merged means the instruction-following layer is built on top of a model that already understands Polish — a significantly better initialization than going from the English-first instruct model directly to Polish instruction data. The downside is the two-stage pipeline takes more time, but the quality improvement is substantial for low-resource language adaptation.

**Why train_on_responses_only=True:** In standard causal language modeling, loss is computed over all tokens including user turns. This is correct for CPT (where there is no distinction between "input" and "output") but wrong for SFT. Learning to predict user messages teaches the model the distribution of user questions, which doesn't help it answer them. Masking user turns to zero loss focuses all gradient signal on producing better assistant responses, which is what evaluation measures.

**Why enable_thinking=False:** Qwen3.6-27B's chat template generates `<think>` tokens before answering by default. For SFT on Polish instructions, we want the model to answer directly without extended thinking chains in the training labels (the Dolci-Instruct and catalog Q&A datasets don't contain thinking traces). Training with thinking enabled would produce `<think>...</think>` blocks in labels that the model must learn to reproduce, which is not the target behavior.

**Why lower LR than CPT (2e-4 but still higher than CPT's 3e-5):** SFT is a smaller distributional shift than CPT — the model's weights are already adapted to Polish and we're now teaching response format and style. 2e-4 is standard for LoRA SFT. The reason it can be higher than CPT's 3e-5 is that CPT was starting from an instruct model whose alignment we wanted to preserve; at SFT time the model is already Polish-fluent and we want the instruction following to update more aggressively.

**Why r=32 (reduced from r=64):** SFT needs less adapter capacity than CPT. The heavy representational shift happened in CPT; SFT is a refinement. r=32 is sufficient for instruction-following adaptation and keeps the adapter smaller, which speeds up training and reduces the risk of overfitting on the 495k SFT examples.

**Why 3 epochs:** The SFT dataset (~500k examples) is much smaller than the CPT corpus. A single epoch would underfit — the model wouldn't generalize instruction-following patterns well. 3 epochs is the standard for LoRA SFT on datasets of this size; more than 3 epochs risks memorizing the template structure rather than generalizing.

```bash
# configs/sft.yaml already points base_model to models/cpt/merged
python scripts/train/sft.py --config configs/sft.yaml
# 3 epochs. Checkpoints → models/sft/

# Merge:
python scripts/train/sft.py --config configs/sft.yaml --merge
# Output: models/sft/merged/
```

> To skip CPT and start directly from the base model (faster iteration),
> edit `configs/sft.yaml` and set `base_model: Qwen/Qwen3.6-27B`.

---

### Step 8 — Preference Optimization (DPO)

**What it does:** Adjusts the model's response style and helpfulness using pairs of human-preferred and less-preferred responses for the same prompt. Operates on the SFT-merged model for one epoch at a very low learning rate.

**Why DPO (not PPO/RLHF):** Direct Preference Optimization eliminates the need for a separate reward model and the complex online RL training loop that PPO requires. In PPO, you train a reward model on preference data, then run rollouts from the policy, score them with the reward model, and compute policy gradients — four interacting components that each introduce failure modes. DPO reformulates the same objective as a direct supervised loss on the preference pairs. Given a single GPU and the size of our DPO dataset (225k pairs), PPO would be impractical; DPO runs in the same framework as SFT.

**Why very low LR (5e-6, 37× lower than SFT):** DPO is extremely sensitive to learning rate. A high LR causes reward hacking: the model learns to produce responses that score well on the preference metric but diverge significantly from the SFT policy, often degrading coherence or instruction following. The beta=0.1 parameter (KL divergence penalty) controls how far the DPO policy can drift from the SFT reference; a low LR keeps the updates small enough that beta has time to act as a brake.

**Why beta=0.1:** Beta is the KL penalty weight in the DPO loss. High beta (>0.5) makes the model barely update — the preference signal is overwhelmed by the penalty for diverging from SFT. Low beta (<0.05) allows the model to overfit to the preference pairs and collapse to a narrow output distribution. 0.1 is the standard from the original DPO paper and works well on datasets of this size.

**Why sigmoid loss (not IPO or KTO):** Sigmoid (standard DPO) is the most well-understood loss and has the most stable training behavior. IPO (Identity Preference Optimization) was proposed to fix theoretical issues with DPO's margin term but introduces different hyperparameter sensitivity. KTO (Kahneman-Tversky Optimization) can work with unpaired preferences, which is irrelevant here since Dolci-DPO provides matched pairs. Sigmoid DPO is the right default for standard paired preference data.

**Why 1 epoch:** DPO datasets are small relative to CPT/SFT corpora. More than 1 epoch on 225k pairs causes reward hacking even at low LR — the model starts optimizing for the training distribution of the Dolci-DPO pairs rather than generalizing to user preferences. One epoch is a well-established default for DPO.

```bash
python scripts/train/dpo.py --config configs/dpo.yaml
# 1 epoch. Checkpoints → models/dpo/

# Merge:
python scripts/train/dpo.py --config configs/dpo.yaml --merge
# Output: models/dpo/merged/
```

---

### Step 9 — Evaluate

**What it does:** Measures the fine-tuned model on three axes: Polish language tasks (fluency, reasoning, knowledge), English retention (catastrophic forgetting check), and closed-book catalog knowledge (did the facts bake in?).

**Why three separate evaluation suites:**

- **Polish suite:** The Open PL LLM Leaderboard benchmarks (PolEmo, KLEJ, Belebele PL, etc.) measure whether the model actually improved at Polish — fluency, reasoning, and cultural knowledge. Without this, you can't distinguish "model is more Polish" from "model is just more verbose in Polish."

- **English retention:** Compares the fine-tuned model against the original Qwen/Qwen3.6-27B baseline on MMLU, HellaSwag, and ARC (English). The goal is that English scores don't drop more than 2–3 percentage points. If they drop more, the English replay fraction in CPT was insufficient.

- **Catalog knowledge (closed-book):** Tests whether facts from legal acts, GUS statistics, and dane.gov.pl descriptions actually entered the model's weights. The holdout question set comes from catalog records deliberately excluded from training (`data/catalogs/_holdout/`). This is the most direct test of the core project goal — knowledge injection without RAG.

```bash
# Polish tasks (Open PL LLM Leaderboard suite, Belebele PL, etc.)
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

# Closed-book catalog knowledge (facts baked into weights)
# First build a held-out question set from catalog records kept out of training:
python scripts/process/build_sft_qa.py \
    --input "data/catalogs/_holdout/**/*.jsonl" \
    --out eval/data/catalog_qa_holdout.jsonl \
    --mode template --per-record 1

python scripts/eval/catalog_eval.py \
    --model models/dpo/merged \
    --qa eval/data/catalog_qa_holdout.jsonl \
    --out eval/results/catalog
```

Results are written to `eval/results/`. Compare the fine-tuned model against
`Qwen/Qwen3.6-27B` (the untuned baseline) on all Polish tasks.

---

### Step 10 — Export to GGUF

**What it does:** Converts the merged DPO model to GGUF format and quantizes it to multiple bit-widths. The GGUF files are the final deliverable — self-contained model files that run in llama.cpp and Ollama without any Python dependencies.

**Why GGUF (not safetensors or ONNX):** GGUF is the universal format for llama.cpp and Ollama — the dominant serving stacks for self-hosted models. It embeds the tokenizer, chat template, and quantized weights in a single file, eliminating the need for a transformers install at inference time. ONNX is an alternative but requires ONNX Runtime, has weaker support for quantization at this scale, and doesn't natively support Qwen's SSM-hybrid architecture. Safetensors is the training format — keeping it means keeping the full bf16 model (~54 GB), which is impractical to distribute.

**Why multiple quantization levels:**

| Quant | Size | Use case |
|---|---|---|
| `Q4_K_M` | ~17 GB | Primary deployment — best quality/size tradeoff; fits in 24 GB VRAM |
| `Q5_K_M` | ~20 GB | Higher quality deployment when VRAM allows |
| `Q6_K` | ~23 GB | Near-lossless; for quality-sensitive applications |
| `Q8_0` | ~28 GB | Reference quality — used to verify quantization loss vs. f16 |

**Why two export backends:** Unsloth has its own GGUF exporter that runs within the same Python session as training, which is faster and avoids a separate llama.cpp compile. However, as of planning time, Unsloth has not confirmed `FastQwen3_5Model` support for the SSM-hybrid architecture. The llama.cpp `convert_hf_to_gguf.py` approach always works regardless of architecture and is the safe fallback.

```bash
# Preferred: Unsloth one-call exporter (if Qwen3.6 support is available)
python scripts/train/export_gguf.py \
    --config configs/dpo.yaml \
    --backend unsloth \
    --quants Q4_K_M Q5_K_M Q6_K Q8_0 \
    --out models/gguf

# Fallback: llama.cpp exporter (always works)
python scripts/train/export_gguf.py \
    --config configs/dpo.yaml \
    --backend llamacpp \
    --quants Q4_K_M Q5_K_M Q6_K Q8_0 \
    --out models/gguf \
    --llama-dir llama.cpp
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

## Quick reference — all commands in order

```bash
# 0. env (torch==2.10.0 already in vllm venv — skip torch install)
pip install -r requirements.txt
python scripts/check_env.py

# 1. catalogs
python scripts/ingest/sejm_isap.py --publisher DU --years 2015-2024 --out data/catalogs/isap/du.jsonl
python scripts/ingest/dane_gov.py  --out data/catalogs/dane_gov/datasets.jsonl --commercial-safe
python scripts/ingest/gus_bdl.py   --list-subjects  # discover valid IDs first
python scripts/ingest/gus_bdl.py   --subjects K11,K15,K27 --years 2010-2023 --out data/catalogs/gus_bdl/indicators.jsonl

# 2. corpora  (HF download — see Step 2 above)

# 3. process
python scripts/process/pipeline.py --input "data/raw/**/*.jsonl" --output data/interim/clean --workers 16
python scripts/process/dedup.py    --input data/interim/clean --output data/interim/dedup --workers 16

# 4. datasets
python scripts/process/build_cpt_mix.py --pl "data/interim/dedup/**/*.jsonl" "data/catalogs/**/*.jsonl" --en "data/raw/replay_en/**/*.jsonl" --out data/processed/cpt --commercial-safe
python scripts/process/build_sft_qa.py  --input "data/catalogs/**/*.jsonl" --out data/processed/sft/catalog_qa.jsonl

# 5. pilot (small run — confirm stack works)
python scripts/train/cpt.py --config configs/cpt.yaml   # with max_steps: 200

# 6-8. train
python scripts/train/cpt.py --config configs/cpt.yaml && python scripts/train/cpt.py --config configs/cpt.yaml --merge
python scripts/train/sft.py --config configs/sft.yaml && python scripts/train/sft.py --config configs/sft.yaml --merge
python scripts/train/dpo.py --config configs/dpo.yaml && python scripts/train/dpo.py --config configs/dpo.yaml --merge

# 9. eval
python scripts/eval/run_eval.py --model models/dpo/merged --suite polish
python scripts/eval/run_eval.py --model models/dpo/merged --suite english
python scripts/eval/catalog_eval.py --model models/dpo/merged --qa eval/data/catalog_qa_holdout.jsonl

# 10. export
python scripts/train/export_gguf.py --config configs/dpo.yaml --quants Q4_K_M Q5_K_M Q8_0 --out models/gguf
python scripts/eval/smoke_gguf.py   --gguf models/gguf/model-Q4_K_M.gguf
```

---

## Project layout

```
configs/          cpt.yaml  sft.yaml  dpo.yaml
scripts/
  check_env.py
  ingest/         sejm_isap.py  dane_gov.py  gus_bdl.py  culturax_pl.py
  process/        pipeline.py  dedup.py  build_cpt_mix.py  build_sft_qa.py  quality_pl.py
  train/          cpt.py  sft.py  dpo.py  export_gguf.py  _common.py
  eval/           run_eval.py  catalog_eval.py  smoke_gguf.py
data/
  raw/            downloaded corpora (gitignored)
  interim/        clean/  dedup/  (gitignored)
  processed/      cpt/  sft/  dpo/  (gitignored)
  catalogs/       ISAP  dane.gov.pl  GUS BDL  (gitignored)
models/           adapters  merged  gguf  (gitignored)
eval/results/     benchmark scores  (gitignored)
docs/             SETUP.md
plans/            implementation plan
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check_env.py` bf16 test fails | PyTorch lacks `sm_120` kernels — reinstall from `cu128` index |
| `Unsloth unavailable` in training log | Expected for Qwen3.6 (new arch); PEFT fallback is used automatically |
| OOM during CPT | Reduce `per_device_train_batch_size` to 1, double `gradient_accumulation_steps` |
| `target_modules not found` PEFT error | Run model + `print([n for n,_ in model.named_modules()])` to check layer names |
| GGUF smoke test produces garbled text | Chat template mismatch — ensure `<\|im_start\|>` tokens are in the GGUF's tokenizer |
| lm-eval task not found | Run `lm-eval --tasks list \| grep -i pl` to get current Polish task IDs |
| English scores drop >3% after CPT | Increase `replay_fraction` in `build_cpt_mix.py` (try 0.22), rebuild CPT mix, restart CPT |
| DPO loss spikes or diverges | LR is too high — halve `learning_rate` in `configs/dpo.yaml` and restart from SFT merged |
