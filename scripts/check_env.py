#!/usr/bin/env python3
"""Verify the training environment is usable.

Run this FIRST, before any training:
    python scripts/check_env.py

Defaults target the reference rig (RTX 6000 Pro Blackwell, sm_120, ~96 GB), but the
thresholds are configurable so the check works on any GPU:
    python scripts/check_env.py --min-vram 24 --min-compute 8.0

Multi-GPU rigs are checked device by device. --min-vram is per device by default
(QLoRA loads the whole model on one card); pass --vram-total if you shard the model
across GPUs and only care about the aggregate:
    python scripts/check_env.py --vram-total

Checks:
  - torch + CUDA versions
  - per-GPU name, compute capability, VRAM (warns below thresholds)
  - a tiny bf16 matmul actually runs on every GPU (confirms kernels exist for each arch)
  - mixed-GPU rigs are flagged (different models/capabilities break naive sharding)
  - NCCL availability when more than one GPU is visible
  - presence of key libraries (training + eval/export backends)
"""

from __future__ import annotations

import argparse
import importlib
import os
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
    ap.add_argument("--vram-total", action="store_true",
                    help="apply --min-vram to the sum across GPUs instead of per device "
                         "(only meaningful if the model is sharded across GPUs)")
    ap.add_argument("--min-gpus", type=int, default=1,
                    help="fail if fewer than this many GPUs are visible (default 1)")
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

    n_gpus = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"\n== GPUs ({n_gpus} visible) ==")
    if visible is not None:
        print(f"  CUDA_VISIBLE_DEVICES={visible}")
    if n_gpus < args.min_gpus:
        print(f"\nFATAL: {n_gpus} GPU(s) visible, --min-gpus {args.min_gpus} required.")
        return 1

    names: list[str] = []
    caps: list[tuple[int, int]] = []
    total_all_gb = 0.0
    for idx in range(n_gpus):
        props = torch.cuda.get_device_properties(idx)
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        total_gb = props.total_memory / (1024 ** 3)
        names.append(name)
        caps.append(cap)
        total_all_gb += total_gb

        print(f"\n  -- GPU {idx} --")
        print(f"  name                {name}")
        print(f"  compute capability  sm_{cap[0]}{cap[1]} ({cap[0]}.{cap[1]})")
        print(f"  total VRAM          {total_gb:.1f} GB")

        if (cap[0] + cap[1] / 10) < args.min_compute:
            print(f"  WARNING: compute capability {cap[0]}.{cap[1]} < {args.min_compute}; "
                  "the reference recipe targets Blackwell (sm_120). bf16/kernels may be "
                  "slow or unsupported.")
        if not args.vram_total and total_gb < args.min_vram:
            print(f"  WARNING: {total_gb:.1f} GB VRAM < {args.min_vram} GB. The 27B QLoRA "
                  "recipe assumes ~96 GB on one card; reduce seq len / batch, shard across "
                  "GPUs (then re-run with --vram-total), or use a smaller model.")

    if n_gpus > 1:
        print(f"\n  total VRAM across GPUs  {total_all_gb:.1f} GB")
        if len(set(names)) > 1:
            print(f"  WARNING: mixed GPU models {sorted(set(names))}. Sharding assumes "
                  "identical devices; the slowest/smallest card gates the run.")
        if len(set(caps)) > 1:
            print("  WARNING: mixed compute capabilities across GPUs; kernels available on "
                  "one device may be missing on another.")
        print(f"  NCCL available      {torch.distributed.is_nccl_available()}")
        if not torch.distributed.is_nccl_available():
            print("  WARNING: NCCL unavailable — multi-GPU training (DDP/FSDP/accelerate) "
                  "will not work; single-GPU runs are unaffected.")

    # Confirm kernels actually run on each architecture present.
    print("\n== bf16 matmul smoke test ==")
    for idx in range(n_gpus):
        try:
            dev = torch.device("cuda", idx)
            a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
            b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
            c = (a @ b).float().sum().item()
            torch.cuda.synchronize(dev)
            print(f"  GPU {idx}: OK — result sum = {c:.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  GPU {idx}: FAILED: {exc}")
            print("  This usually means torch lacks kernels for this device's arch "
                  "(sm_120 on Blackwell). Reinstall a CUDA 12.8+ build (see docs/SETUP.md).")
            return 1

    if args.vram_total and total_all_gb < args.min_vram:
        print(f"\nWARNING: {total_all_gb:.1f} GB total VRAM < {args.min_vram} GB across "
              f"{n_gpus} GPU(s).")

    print("\nEnvironment looks usable. ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
