# Orchestration for the pslab-finetuning pipeline.
# Each target wraps the canonical command so runs are reproducible from one entrypoint
# instead of copy-pasting from the README. Override paths via VAR=value on the CLI.

PY ?= python
CPT_CFG ?= configs/cpt.yaml
SFT_CFG ?= configs/sft.yaml
DPO_CFG ?= configs/dpo.yaml

# Multi-GPU: `make cpt NPROC=4` runs 4-way DDP (one full replica per card). NPROC=1 stays a
# plain single process. Merge and eval targets deliberately keep $(PY) — merging is
# single-process by definition, and eval fans out with CUDA_VISIBLE_DEVICES instead.
NPROC  ?= 1
LAUNCH ?= $(if $(filter 1,$(NPROC)),$(PY),torchrun --standalone --nproc_per_node=$(NPROC))
# Number of GPUs the eval fan-out uses (make eval-parallel).
EVAL_GPUS ?= 4

.PHONY: help env test lint ingest process dedup build-cpt build-sft build-agentic \
        cpt cpt-merge sft sft-merge dpo dpo-merge eval eval-parallel eval-agentic gguf

help:
	@echo "Targets: env test lint | ingest process dedup | build-cpt build-sft build-agentic |"
	@echo "         cpt cpt-merge sft sft-merge dpo dpo-merge | eval eval-parallel eval-agentic gguf"
	@echo "Multi-GPU: append NPROC=4 to a training target (e.g. make cpt NPROC=4)."

# --- dev ---------------------------------------------------------------------
env:
	$(PY) scripts/check_env.py
test:
	pytest
lint:
	ruff check scripts tests

# --- data --------------------------------------------------------------------
ingest:
	$(PY) scripts/ingest/sejm_isap.py --out data/catalogs/isap/du.jsonl
	$(PY) scripts/ingest/dane_gov.py --out data/catalogs/dane_gov/datasets.jsonl --commercial-safe
	$(PY) scripts/ingest/gus_bdl.py --subjects K11,K15,K27 --out data/catalogs/gus_bdl/indicators.jsonl

process:
	$(PY) scripts/process/pipeline.py --input "data/raw/**/*.jsonl" --output data/interim/clean --workers 16
dedup:
	$(PY) scripts/process/dedup.py --input data/interim/clean --output data/interim/dedup --threshold 0.8

build-cpt:
	$(PY) scripts/process/build_cpt_mix.py --pl "data/interim/dedup/**/*.jsonl" \
	  --en "data/raw/replay_en/**/*.jsonl" --out data/processed/cpt --commercial-safe
build-sft:
	$(PY) scripts/process/build_sft_qa.py --input "data/catalogs/**/*.jsonl" \
	  --out data/processed/sft/catalog_qa.jsonl --mode template
build-agentic:
	$(PY) scripts/process/build_sft_qa.py --input "data/catalogs/**/*.jsonl" \
	  --out data/processed/sft/agentic/tool_qa.jsonl --mode agentic

# --- training (merge produces <output_dir>/merged for the next stage) ---------
# Add NPROC=4 for 4-way DDP. Merge targets are always single-process.
cpt:
	$(LAUNCH) scripts/train/cpt.py --config $(CPT_CFG)
cpt-merge:
	$(PY) scripts/train/cpt.py --config $(CPT_CFG) --merge
sft:
	$(LAUNCH) scripts/train/sft.py --config $(SFT_CFG)
sft-merge:
	$(PY) scripts/train/sft.py --config $(SFT_CFG) --merge
dpo:
	$(LAUNCH) scripts/train/dpo.py --config $(DPO_CFG)
dpo-merge:
	$(PY) scripts/train/dpo.py --config $(DPO_CFG) --merge

# --- eval + export -----------------------------------------------------------
eval:
	$(PY) scripts/eval/catalog_eval.py --model models/dpo/merged \
	  --qa eval/data/catalog_qa_holdout.jsonl --scorer hybrid

# Data-parallel eval: one single-GPU worker per card over disjoint slices of the holdout,
# then merge. No inter-GPU traffic, so it scales ~linearly — unlike tensor parallelism,
# which would all-reduce every layer across PCIe on this NVLink-less rig.
eval-parallel:
	@for i in $$(seq 0 $$(($(EVAL_GPUS)-1))); do \
	  CUDA_VISIBLE_DEVICES=$$i $(PY) scripts/eval/catalog_eval.py --model models/dpo/merged \
	    --qa eval/data/catalog_qa_holdout.jsonl --scorer hybrid --device-map cuda:0 \
	    --num-shards $(EVAL_GPUS) --shard $$i & \
	done; wait
	$(PY) scripts/eval/merge_shards.py --out eval/results/catalog

eval-agentic:
	$(PY) scripts/eval/agentic_eval.py --model models/dpo/merged \
	  --qa eval/data/agentic_holdout.jsonl
gguf:
	$(PY) scripts/train/export_gguf.py --config $(DPO_CFG) --quants Q4_K_M Q5_K_M Q8_0
