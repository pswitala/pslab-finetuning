# Orchestration for the pslab-finetuning pipeline.
# Each target wraps the canonical command so runs are reproducible from one entrypoint
# instead of copy-pasting from the README. Override paths via VAR=value on the CLI.

PY ?= python
CPT_CFG ?= configs/cpt.yaml
SFT_CFG ?= configs/sft.yaml
DPO_CFG ?= configs/dpo.yaml

.PHONY: help env test lint ingest process dedup build-cpt build-sft build-agentic \
        cpt cpt-merge sft sft-merge dpo dpo-merge eval eval-agentic gguf

help:
	@echo "Targets: env test lint | ingest process dedup | build-cpt build-sft build-agentic |"
	@echo "         cpt cpt-merge sft sft-merge dpo dpo-merge | eval eval-agentic gguf"

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
cpt:
	$(PY) scripts/train/cpt.py --config $(CPT_CFG)
cpt-merge:
	$(PY) scripts/train/cpt.py --config $(CPT_CFG) --merge
sft:
	$(PY) scripts/train/sft.py --config $(SFT_CFG)
sft-merge:
	$(PY) scripts/train/sft.py --config $(SFT_CFG) --merge
dpo:
	$(PY) scripts/train/dpo.py --config $(DPO_CFG)
dpo-merge:
	$(PY) scripts/train/dpo.py --config $(DPO_CFG) --merge

# --- eval + export -----------------------------------------------------------
eval:
	$(PY) scripts/eval/catalog_eval.py --model models/dpo/merged \
	  --qa eval/data/catalog_qa_holdout.jsonl --scorer hybrid
eval-agentic:
	$(PY) scripts/eval/agentic_eval.py --model models/dpo/merged \
	  --qa eval/data/agentic_holdout.jsonl
gguf:
	$(PY) scripts/train/export_gguf.py --config $(DPO_CFG) --quants Q4_K_M Q5_K_M Q8_0
