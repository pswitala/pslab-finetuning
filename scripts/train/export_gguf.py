#!/usr/bin/env python3
"""Phase 7 — export the final model to GGUF and quantize.

Produces the deliverable: GGUF files runnable in llama.cpp / Ollama. Merges the final
DPO adapter into 16-bit weights, then converts + quantizes to several levels.

Two backends:
  - unsloth  (default): one call, `model.save_pretrained_gguf(...)`, wraps llama.cpp.
  - llamacpp : merge with PEFT, then call llama.cpp's convert + quantize binaries
               (set LLAMA_CPP_DIR). Use this if Unsloth GGUF export is unavailable.

Quants: Q4_K_M (default best size/quality), Q5_K_M, Q6_K, Q8_0, and optional f16 ref.

Usage:
    python scripts/train/export_gguf.py --config configs/dpo.yaml \
        --quants Q4_K_M Q5_K_M Q8_0 --out models/gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_config, load_model_and_tokenizer, merge_and_save  # noqa: E402


def export_unsloth(cfg: dict, quants: list[str], out_dir: str) -> None:
    loaded = load_model_and_tokenizer(cfg)
    if loaded.backend != "unsloth":
        raise RuntimeError("Unsloth backend unavailable; use --backend llamacpp")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for q in quants:
        # Unsloth quant names are lowercase, e.g. "q4_k_m".
        method = q.lower()
        print(f"[gguf] exporting {q} -> {out_dir}")
        loaded.model.save_pretrained_gguf(out_dir, loaded.tokenizer,
                                          quantization_method=method)


def export_llamacpp(cfg: dict, quants: list[str], out_dir: str,
                    llama_dir: str) -> None:
    loaded = load_model_and_tokenizer(cfg)
    merged = merge_and_save(loaded, cfg["output_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    llama = Path(llama_dir)

    # 1) HF -> GGUF f16
    f16 = str(Path(out_dir) / "model-f16.gguf")
    convert = llama / "convert_hf_to_gguf.py"
    print(f"[gguf] converting merged -> {f16}")
    subprocess.run([sys.executable, str(convert), merged,
                    "--outfile", f16, "--outtype", "f16"], check=True)

    # 2) quantize f16 -> each level
    quantize_bin = llama / "build" / "bin" / "llama-quantize"
    for q in quants:
        out_q = str(Path(out_dir) / f"model-{q}.gguf")
        print(f"[gguf] quantizing -> {out_q}")
        subprocess.run([str(quantize_bin), f16, out_q, q], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml",
                    help="config whose base_model points at the final checkpoint")
    ap.add_argument("--backend", choices=["unsloth", "llamacpp"], default="unsloth")
    ap.add_argument("--quants", nargs="+",
                    default=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
    ap.add_argument("--out", default="models/gguf")
    ap.add_argument("--llama-dir", default="llama.cpp")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.backend == "unsloth":
        export_unsloth(cfg, args.quants, args.out)
    else:
        export_llamacpp(cfg, args.quants, args.out, args.llama_dir)

    print(f"\n[gguf] done -> {args.out}")
    print("Next: smoke-test with scripts/eval/smoke_gguf.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
