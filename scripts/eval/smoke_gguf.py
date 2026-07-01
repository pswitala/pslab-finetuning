#!/usr/bin/env python3
"""Smoke-test an exported GGUF: load it and run a few Polish prompts.

Confirms the quantized model loads, uses the correct chat template, and generates
sensible Polish. Prefers llama-cpp-python; falls back to invoking the llama.cpp CLI.

Usage:
    python scripts/eval/smoke_gguf.py --gguf models/gguf/model-Q4_K_M.gguf
    python scripts/eval/smoke_gguf.py --gguf models/gguf/model-Q4_K_M.gguf \
        --llama-cli llama.cpp/build/bin/llama-cli
"""

from __future__ import annotations

import argparse
import subprocess
import sys

PROMPTS = [
    "Wyjaśnij krótko, czym jest Konstytucja Rzeczypospolitej Polskiej.",
    "Wymień trzy największe miasta w Polsce i krótko je opisz.",
    "Czym zajmuje się Główny Urząd Statystyczny (GUS)?",
    "Napisz dwa zdania o Mikołaju Koperniku.",
]


def via_python(gguf: str, n_predict: int) -> bool:
    try:
        from llama_cpp import Llama
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] llama-cpp-python unavailable ({exc}); trying CLI")
        return False
    llm = Llama(model_path=gguf, n_ctx=4096, n_gpu_layers=-1, verbose=False)
    for p in PROMPTS:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": p}],
            max_tokens=n_predict, temperature=0.7)
        text = out["choices"][0]["message"]["content"].strip()
        print(f"\nQ: {p}\nA: {text}\n{'-' * 60}")
    return True


def via_cli(gguf: str, llama_cli: str, n_predict: int) -> bool:
    for p in PROMPTS:
        cmd = [llama_cli, "-m", gguf, "-p", p, "-n", str(n_predict), "-ngl", "999"]
        print(f"\nQ: {p}")
        subprocess.run(cmd, check=False)
        print("-" * 60)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-cli", default="llama.cpp/build/bin/llama-cli")
    ap.add_argument("--n-predict", type=int, default=200)
    args = ap.parse_args()

    if not via_python(args.gguf, args.n_predict):
        via_cli(args.gguf, args.llama_cli, args.n_predict)
    print("\n[smoke] done — eyeball the Polish answers above for fluency/correctness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
