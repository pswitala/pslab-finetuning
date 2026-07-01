#!/usr/bin/env python3
"""Closed-book catalog-knowledge eval.

Measures how well catalog facts that were baked into the weights are recalled WITHOUT
retrieval. Build the question set from held-out catalog records (kept out of training),
then score the model's free-text answers against reference facts.

Scoring: lightweight string/number-overlap by default; plug an LLM judge for nuance.
Compare the fine-tuned model against the base model (which should be near-zero on
Polish-specific public data).

Build a held-out set first (questions derived from records, with reference answers):
    python scripts/process/build_sft_qa.py --input "data/catalogs/_holdout/**/*.jsonl" \
        --out eval/data/catalog_qa_holdout.jsonl --mode template --per-record 1

Then evaluate:
    python scripts/eval/catalog_eval.py --model models/dpo/merged \
        --qa eval/data/catalog_qa_holdout.jsonl --out eval/results/catalog

    # vllm backend (batches all prompts at once — faster throughput):
    python scripts/eval/catalog_eval.py --model models/dpo/merged \
        --qa eval/data/catalog_qa_holdout.jsonl --backend vllm

    # GGUF backend (--model is the .gguf file path):
    python scripts/eval/catalog_eval.py --model models/gguf/model-Q4_K_M.gguf \
        --qa eval/data/catalog_qa_holdout.jsonl --backend gguf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def normalize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower(), re.UNICODE))


def overlap_score(answer: str, reference: str) -> float:
    a, r = normalize(answer), normalize(reference)
    if not r:
        return 0.0
    return len(a & r) / len(r)


def load_qa(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex["messages"]
            q = next(m["content"] for m in msgs if m["role"] == "user")
            ref = next(m["content"] for m in msgs if m["role"] == "assistant")
            items.append({"q": q, "ref": ref})
    return items


def _apply_chat_template(tok, question: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True)


def _infer_hf(args, qa: list[dict]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    model.eval()

    answers = []
    for ex in qa:
        prompt = _apply_chat_template(tok, ex["q"])
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        answer = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
        answers.append(answer)
    return answers


def _infer_vllm(args, qa: list[dict]) -> list[str]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
    )
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    prompts = [_apply_chat_template(tok, ex["q"]) for ex in qa]
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def _infer_gguf(args, qa: list[dict]) -> list[str]:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            f"[catalog_eval] llama-cpp-python not available: {exc}\n"
            "Install with GPU support: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
        ) from exc

    llm = Llama(model_path=args.model, n_ctx=4096, n_gpu_layers=-1, verbose=False)
    answers = []
    for ex in qa:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": ex["q"]}],
            max_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        answers.append(out["choices"][0]["message"]["content"].strip())
    return answers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model path/id, or .gguf file path when --backend gguf")
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", default="eval/results/catalog")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="overlap >= threshold counts as recalled")
    ap.add_argument("--backend", choices=["hf", "vllm", "gguf"], default="hf",
                    help="Inference backend: hf (default), vllm, or gguf")
    args = ap.parse_args()

    qa = load_qa(args.qa)
    if not qa:
        print("[catalog_eval] no QA items found; check --qa path")
        return 1

    if args.backend == "hf":
        answers = _infer_hf(args, qa)
    elif args.backend == "vllm":
        answers = _infer_vllm(args, qa)
    else:
        answers = _infer_gguf(args, qa)

    results, hits, total_overlap = [], 0, 0.0
    for ex, answer in zip(qa, answers):
        sc = overlap_score(answer, ex["ref"])
        hits += sc >= args.threshold
        total_overlap += sc
        results.append({"q": ex["q"], "ref": ex["ref"], "answer": answer, "score": sc})

    n = len(qa)
    summary = {
        "n": n,
        "recall_rate": hits / n if n else 0.0,
        "mean_overlap": total_overlap / n if n else 0.0,
        "model": args.model,
        "backend": args.backend,
        "threshold": args.threshold,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
