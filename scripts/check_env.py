#!/usr/bin/env python3
"""Verify the training environment is usable on the Blackwell GPU.

Run this FIRST, before any training:
    python scripts/check_env.py

Checks:
  - torch + CUDA versions
  - GPU name, compute capability (expect sm_120 / 12.0 for RTX 6000 Pro Blackwell)
  - available VRAM (expect ~96 GB)
  - a tiny bf16 matmul actually runs on the GPU (confirms kernels exist for sm_120)
  - presence of key libraries (unsloth, transformers, trl, peft, bitsandbytes)
"""

from __future__ import annotations

import importlib
import sys


def _check_lib(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({exc.__class__.__name__})"


def main() -> int:
    print("== Library versions ==")
    for lib in ("torch", "transformers", "trl", "peft", "bitsandbytes", "unsloth",
                "datasets", "accelerate", "datatrove", "lm_eval"):
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

    if cap[0] < 9:
        print("  WARNING: pre-Hopper GPU; this project targets Blackwell (sm_120).")
    if total_gb < 70:
        print("  WARNING: < 70 GB VRAM. The 27B QLoRA recipe assumes ~96 GB; "
              "reduce seq len / batch or use a smaller model.")

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
