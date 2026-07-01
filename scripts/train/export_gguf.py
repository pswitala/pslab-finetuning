#!/usr/bin/env python3
"""Phase 7 — export the final model to GGUF and quantize.

Produces the deliverable: GGUF files runnable in llama.cpp / Ollama.

IMPORTANT: this step operates on an already-MERGED checkpoint (LoRA folded into the
base weights), NOT on a config with a raw adapter. Run the final training stage with
`--merge` first, e.g.:

    python scripts/train/dpo.py --config configs/dpo.yaml --merge   # -> models/dpo/merged
    python scripts/train/export_gguf.py --config configs/dpo.yaml   # exports models/dpo/merged

The merged dir is resolved as `<config output_dir>/merged` (override with --model-dir).
Earlier versions loaded the config's base_model and attached a FRESH, UNTRAINED adapter,
which silently exported the wrong (pre-fine-tune) weights — this version reads the trained,
merged checkpoint directly so the export always reflects your training.

Two backends:
  - unsloth  (default): loads the merged checkpoint via Unsloth and calls
               `model.save_pretrained_gguf(...)` (wraps llama.cpp).
  - llamacpp : converts the merged HF checkpoint with llama.cpp's convert + quantize
               binaries (set --llama-dir). Use if Unsloth GGUF export is unavailable.

Quants: Q4_K_M (default best size/quality), Q5_K_M, Q6_K, Q8_0, and optional f16 ref.

Usage:
    python scripts/train/export_gguf.py --config configs/dpo.yaml \
        --quants Q4_K_M Q5_K_M Q8_0 --out models/gguf
    python scripts/train/export_gguf.py --model-dir models/dpo/merged --backend llamacpp
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_config  # noqa: E402


def _resolve_model_dir(args) -> str:
    """The merged checkpoint to export: --model-dir, else <config output_dir>/merged."""
    if args.model_dir:
        model_dir = args.model_dir
    else:
        cfg = load_config(args.config)
        model_dir = str(Path(cfg["output_dir"]) / "merged")
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(
            f"merged checkpoint not found: {model_dir}\n"
            f"Run the final training stage with --merge first "
            f"(e.g. `python scripts/train/dpo.py --config {args.config} --merge`)."
        )
    return model_dir


def export_unsloth(model_dir: str, quants: list[str], out_dir: str) -> None:
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unsloth unavailable ({exc}); use --backend llamacpp") from exc
    import torch
    # Load the already-merged weights as-is (no adapter attached).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir, load_in_4bit=False, dtype=torch.bfloat16)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for q in quants:
        # Unsloth quant names are lowercase, e.g. "q4_k_m".
        print(f"[gguf] exporting {q} -> {out_dir}")
        model.save_pretrained_gguf(out_dir, tokenizer, quantization_method=q.lower())


def export_llamacpp(model_dir: str, quants: list[str], out_dir: str,
                    llama_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    llama = Path(llama_dir)

    # 1) HF (merged) -> GGUF f16
    f16 = str(Path(out_dir) / "model-f16.gguf")
    convert = llama / "convert_hf_to_gguf.py"
    print(f"[gguf] converting {model_dir} -> {f16}")
    subprocess.run([sys.executable, str(convert), model_dir,
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
                    help="config whose output_dir holds the merged/ checkpoint to export")
    ap.add_argument("--model-dir", default=None,
                    help="explicit merged checkpoint dir (overrides --config)")
    ap.add_argument("--backend", choices=["unsloth", "llamacpp"], default="unsloth")
    ap.add_argument("--quants", nargs="+",
                    default=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
    ap.add_argument("--out", default="models/gguf")
    ap.add_argument("--llama-dir", default="llama.cpp")
    args = ap.parse_args()

    model_dir = _resolve_model_dir(args)
    print(f"[gguf] exporting merged checkpoint: {model_dir}")

    if args.backend == "unsloth":
        export_unsloth(model_dir, args.quants, args.out)
    else:
        export_llamacpp(model_dir, args.quants, args.out, args.llama_dir)

    print(f"\n[gguf] done -> {args.out}")
    print("Next: smoke-test with scripts/eval/smoke_gguf.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
