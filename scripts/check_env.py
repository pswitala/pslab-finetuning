#!/usr/bin/env python3
"""Verify the training environment is usable.

Run this FIRST, before any training:
    python scripts/check_env.py

Defaults target the reference rig (RTX 6000 Pro Blackwell, sm_120, ~96 GB), but the
thresholds are configurable so the check works on any GPU:
    python scripts/check_env.py --min-vram 24 --min-compute 8.0

Checks:
  - torch + CUDA versions
  - GPU name, compute capability, available VRAM (warns below thresholds)
  - a tiny bf16 matmul actually runs on the GPU (confirms kernels exist for this arch)
  - presence of key libraries (training + eval/export backends)
"""

from __future__ import annotations

import argparse
import importlib
import sys


def _check_lib(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({exc.__class__.__name__})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-vram", type=float, default=70.0,
                    help="warn if total VRAM is below this many GB (default 70)")
    ap.add_argument("--min-compute", type=float, default=9.0,
                    help="warn if compute capability is below this (default 9.0 = Hopper)")
    args = ap.parse_args()

    print("== Library versions ==")
    # Training libs are required; eval/export backends (vllm, llama_cpp) are optional.
    for lib in ("torch", "transformers", "trl", "peft", "bitsandbytes", "unsloth",
                "datasets", "accelerate", "datatrove", "lm_eval", "vllm", "llama_cpp"):
        print(f"  {lib:14s} {_check_lib(lib)}")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"\nFATAL: torch not importable: {exc}")
        return 1

    print("\n== CUDA ==")
    print(f"  torch.__version__   {torch.__version__}")
    print(f"  torch.version.cuda  {torch.version.cuda}")
    print(f"  cuda available      {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nFATAL: CUDA not available. Install a CUDA 12.8+ build of torch "
              "with sm_120 support (see docs/SETUP.md).")
        return 1

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    print(f"\n== GPU {idx} ==")
    print(f"  name                {name}")
    print(f"  compute capability  sm_{cap[0]}{cap[1]} ({cap[0]}.{cap[1]})")
    print(f"  total VRAM          {total_gb:.1f} GB")

    if (cap[0] + cap[1] / 10) < args.min_compute:
        print(f"  WARNING: compute capability {cap[0]}.{cap[1]} < {args.min_compute}; "
              "the reference recipe targets Blackwell (sm_120). bf16/kernels may be slow "
              "or unsupported.")
    if total_gb < args.min_vram:
        print(f"  WARNING: {total_gb:.1f} GB VRAM < {args.min_vram} GB. The 27B QLoRA "
              "recipe assumes ~96 GB; reduce seq len / batch or use a smaller model.")

    # Confirm kernels actually run for this architecture.
    print("\n== bf16 matmul smoke test ==")
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        c = (a @ b).float().sum().item()
        torch.cuda.synchronize()
        print(f"  OK — result sum = {c:.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        print("  This usually means torch lacks sm_120 kernels. Reinstall a "
              "Blackwell-compatible CUDA 12.8+ build (see docs/SETUP.md).")
        return 1

    print("\nEnvironment looks usable. ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
