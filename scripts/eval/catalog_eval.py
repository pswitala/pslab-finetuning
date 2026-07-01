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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", default="eval/results/catalog")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="overlap >= threshold counts as recalled")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Try CausalLM first; VLMs fall back to Vision2Seq.
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

    qa = load_qa(args.qa)
    results, hits, total_overlap = [], 0, 0.0
    for ex in qa:
        # Disable thinking mode for clean closed-book answers.
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": ex["q"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": ex["q"]}],
                tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        answer = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
        sc = overlap_score(answer, ex["ref"])
        hits += sc >= args.threshold
        total_overlap += sc
        results.append({"q": ex["q"], "ref": ex["ref"], "answer": answer, "score": sc})

    n = len(qa)
    summary = {"n": n, "recall_rate": hits / n if n else 0.0,
               "mean_overlap": total_overlap / n if n else 0.0,
               "model": args.model, "threshold": args.threshold}
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
