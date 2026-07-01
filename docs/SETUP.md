# Setup — Ubuntu, RTX 6000 Pro Blackwell (96 GB)

Target: **Ubuntu**, single **NVIDIA RTX 6000 Pro Blackwell**, compute capability
`sm_120`, CUDA 12.9, running inside the **`vllm` virtualenv**.

---

## What is already in the vllm venv

The following packages are **already installed** — do not reinstall them:

```
torch==2.10.0              (with CUDA 12.9 nvidia packages)
transformers==5.5.0
tokenizers==0.22.2
safetensors==0.7.0
huggingface_hub==1.9.0
sentencepiece==0.2.1
protobuf==6.33.6
numpy==2.2.6
tqdm==4.67.3
PyYAML==6.0.3
requests==2.33.1
python-dotenv==1.2.2
regex==2026.4.4

vllm==0.19.0               (for fast inference + eval)
flashinfer-python==0.6.6   (Blackwell-compatible attention kernels — replaces flash-attn)
gguf==0.18.0               (GGUF library — used by export_gguf.py)
triton==3.6.0
ninja==1.13.0
cuda-bindings==12.9.4
cuda-python==12.9.4
```

`flashinfer` is already Blackwell-compatible. **Do not install `flash-attn`** —
it is redundant and will likely fail to build for `sm_120`.

---

## 1. Verify CUDA and GPU

```bash
nvidia-smi                  # confirm driver + GPU name (RTX 6000 Pro Blackwell)
nvcc --version              # confirm CUDA toolkit >= 12.8

# compute capability check:
python3 -c "import torch; print(torch.cuda.get_device_capability())"
# expect: (12, 0)  for sm_120

# VRAM check:
python3 -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
# expect: ~96.0 GB
```

---

## 2. Install missing training packages

PyTorch is **already installed**. Only install what is missing:

```bash
# Make sure you are in the vllm venv first:
# source /path/to/vllm-venv/bin/activate

pip install -r requirements.txt
```

This installs only the packages that are missing from the vllm venv:

| Package | Purpose |
|---|---|
| `peft` | LoRA / rsLoRA / DoRA adapters |
| `trl` | SFTTrainer / DPOTrainer |
| `accelerate` | training backend (bf16, gradient checkpointing) |
| `bitsandbytes` | 4-bit NF4 quantization for QLoRA |
| `datasets` | HF dataset loading (not in vllm venv) |
| `pyarrow` | parquet shards for CPT mixture |
| `zstandard` | decompression (datatrove dependency) |
| `datatrove` | corpus filtering + MinHash dedup pipeline |
| `trafilatura` | HTML → clean text (Sejm/ISAP ingest) |
| `lm-eval` | Polish + English benchmark evaluation |

If `bitsandbytes` reports a CUDA error on first import, upgrade it:

```bash
pip install bitsandbytes --upgrade
```

---

## 3. Install optional components

### Unsloth (faster QLoRA kernels, optional)

```bash
pip install unsloth
```

**Note:** Unsloth's `FastQwen3_5Model` support for Qwen3.6's SSM-hybrid
(`qwen3_5` arch) may not be available yet. If loading fails, `_common.py`
automatically falls back to plain PEFT — no action required. Check
https://github.com/unslothai/unsloth/releases for a Qwen3.6 entry.

### fastText language-ID model

```bash
pip install fasttext-wheel
mkdir -p data/models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -O data/models/lid.176.bin
```

### KenLM (perplexity filtering, optional)

Requires a C++ build toolchain:

```bash
sudo apt install build-essential libboost-all-dev cmake
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### llama.cpp (GGUF export fallback)

Only needed if Unsloth's built-in `save_pretrained_gguf` is unavailable.
`gguf==0.18.0` is already installed in the venv; this adds the CLI binaries:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j$(nproc)
```

---

## 4. Hugging Face authentication

```bash
huggingface-cli login        # enter your HF token when prompted
# or set in .env:
echo "HF_TOKEN=hf_xxxx" >> .env
```

---

## 5. Make scripts executable

```bash
chmod +x scripts/check_env.py scripts/**/*.py
```

---

## 6. Final verification

```bash
python3 scripts/check_env.py
# Expected output:
#   torch.__version__         2.10.0+cu129  (or similar)
#   cuda available            True
#   name                      NVIDIA RTX 6000 Pro Blackwell
#   compute capability        sm_120 (12.0)
#   total VRAM                ~96.0 GB
#   bf16 matmul smoke test    OK

python3 scripts/process/quality_pl.py   # offline self-test, no GPU needed
```

Both should complete without errors.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `sm_120` bf16 test fails | `torch==2.10.0` should support sm_120; if not, reinstall from `--index-url https://download.pytorch.org/whl/cu129` |
| `bitsandbytes` CUDA error on Blackwell | `pip install bitsandbytes --upgrade` |
| `flash-attn` not found error | Do not install flash-attn — `flashinfer` is already present and handles this |
| Unsloth `FastLanguageModel` fails on Qwen3.6 | Expected — PEFT fallback is automatic |
| `datatrove` import error | `pip install datatrove --upgrade` |
| `datasets` not found | `pip install datasets pyarrow` (not in the vllm venv by default) |
| OOM during training | Reduce `per_device_train_batch_size` to 1 in the config |
| `target_modules not found` PEFT error | `python3 -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('...', load_in_4bit=True, trust_remote_code=True); print([n for n,_ in m.named_modules()])"` |
