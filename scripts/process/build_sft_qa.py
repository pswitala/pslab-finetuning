#!/usr/bin/env python3
"""Generate Polish instruction (SFT) Q&A pairs grounded in catalog records.

Since there is NO retrieval layer, baked-in catalog knowledge enters the model only
via (a) raw catalog text in CPT and (b) the synthetic QA produced here for SFT.
Generate densely over the facts that matter (legal acts, GUS statistics, definitions).

Two modes:
  - template  (default): deterministic, offline, no LLM. Cheap and license-clean.
  - llm       : call a teacher model to paraphrase/diversify questions per record.
               (hook left as a TODO — wire to your provider; keep outputs auditable.)

Output: jsonl of chat-format examples:
    {"messages": [{"role":"user","content": Q}, {"role":"assistant","content": A}],
     "source": ..., "license": ..., "snapshot_date": ...}

Usage:
    python scripts/process/build_sft_qa.py \
        --input "data/catalogs/**/*.jsonl" \
        --out data/processed/sft/catalog_qa.jsonl \
        --mode template --per-record 2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Per-source question templates. {title}/{text} filled from the record.
TEMPLATES = {
    "gus_bdl": [
        "Jaka była wartość wskaźnika „{title}” według danych GUS?",
        "Podaj statystykę GUS dotyczącą: {title}.",
    ],
    "isap": [
        "Czego dotyczy akt prawny „{title}”?",
        "Streść w kilku zdaniach polski akt prawny: {title}.",
    ],
    "dane.gov.pl": [
        "Opisz zbiór danych „{title}” z portalu dane.gov.pl.",
        "Co zawiera zbiór danych „{title}”?",
    ],
    "_default": [
        "Wyjaśnij: {title}.",
        "Co wiesz na temat: {title}?",
    ],
}


def truncate(text: str, max_chars: int = 1200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "…"


def template_pairs(rec: dict, per_record: int) -> list[dict]:
    source = rec.get("source", "_default")
    title = (rec.get("title") or "").strip()
    text = (rec.get("text") or "").strip()
    if not text:
        return []
    qs = TEMPLATES.get(source, TEMPLATES["_default"])[:per_record]
    answer = truncate(text)
    out = []
    for q in qs:
        out.append({
            "messages": [
                {"role": "user", "content": q.format(title=title or "ten zbiór")},
                {"role": "assistant", "content": answer},
            ],
            "source": source,
            "license": rec.get("license", "unknown"),
            "snapshot_date": rec.get("snapshot_date", ""),
        })
    return out


def llm_pairs(rec: dict, per_record: int) -> list[dict]:
    # TODO: wire to a teacher model to diversify questions / write fuller answers
    # grounded strictly in rec["text"]. Keep prompts + outputs logged for audit.
    raise NotImplementedError("LLM mode not wired yet — use --mode template for now.")


def iter_jsonl(pattern: str):
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help='glob, e.g. "data/catalogs/**/*.jsonl"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--per-record", type=int, default=2)
    args = ap.parse_args()

    gen = template_pairs if args.mode == "template" else llm_pairs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in iter_jsonl(args.input):
            for pair in gen(rec, args.per_record):
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} SFT QA pairs -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
