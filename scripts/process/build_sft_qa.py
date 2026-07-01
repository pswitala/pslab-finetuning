#!/usr/bin/env python3
"""Generate Polish instruction (SFT) Q&A pairs grounded in catalog records.

Since there is NO retrieval layer, baked-in catalog knowledge enters the model only
via (a) raw catalog text in CPT and (b) the synthetic QA produced here for SFT.
Generate densely over the facts that matter (legal acts, GUS statistics, definitions).

Three modes:
  - template  (default): deterministic, offline, no LLM. Cheap and license-clean.
                Plain Q->A pairs (answer = record text).
  - agentic   : deterministic tool-use trajectories grounded in the real APIs the
                records came from (GUS BDL / dane.gov.pl / ISAP). Emits assistant
                `tool_calls` + a `role:"tool"` result + a final grounded answer, plus the
                `tools` schema. See scripts/common/tool_catalog.py + tooling.py.
  - llm       : call a teacher model to paraphrase/diversify questions per record.
               (hook left as a TODO — wire to your provider; keep outputs auditable.)

Output: jsonl of chat-format examples. template/llm:
    {"messages": [{"role":"user","content": Q}, {"role":"assistant","content": A}],
     "source": ..., "license": ..., "snapshot_date": ...}
agentic adds `tool_calls`, a `role:"tool"` turn, and a top-level `tools` list.

Usage:
    python scripts/process/build_sft_qa.py \
        --input "data/catalogs/**/*.jsonl" \
        --out data/processed/sft/catalog_qa.jsonl \
        --mode template --per-record 2

    # tool-use trajectories:
    python scripts/process/build_sft_qa.py \
        --input "data/catalogs/**/*.jsonl" \
        --out data/processed/sft/agentic/tool_qa.jsonl \
        --mode agentic --per-record 2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tool_catalog import ALL_TOOLS, tool_for_source  # noqa: E402
from common.tooling import make_tool_sample, validate_sample  # noqa: E402

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


# Per-source agentic question phrasings. {title}/{unit}/{year}/{publisher}/{pos} filled
# from the record + its meta.
_AGENTIC_QUESTIONS = {
    "gus_bdl": [
        "Ile według GUS wyniósł wskaźnik „{title}” dla jednostki „{unit}” w {year} roku?",
        "Podaj wartość wskaźnika „{title}” ({unit}, {year}) na podstawie danych GUS.",
    ],
    "isap": [
        "Czego dotyczy akt prawny „{title}” ({publisher} {year} poz. {pos})?",
        "Streść polski akt prawny opublikowany jako {publisher} {year} poz. {pos}.",
    ],
    "dane.gov.pl": [
        "Znajdź w portalu dane.gov.pl zbiór danych „{title}”. Co zawiera?",
        "Opisz zbiór danych „{title}” dostępny na dane.gov.pl.",
    ],
}


def _agentic_call(rec: dict):
    """Derive (arguments, tool_result, final_answer) for a record from its meta.

    Returns None if the record lacks the fields its tool needs.
    """
    source = rec.get("source", "")
    tool_name = tool_for_source(source)
    if tool_name is None:
        return None
    meta = rec.get("meta", {}) or {}
    title = (rec.get("title") or "").strip()
    text = (rec.get("text") or "").strip()

    if source == "gus_bdl":
        year, var_id = meta.get("year"), meta.get("var_id")
        if year is None or not var_id:
            return None
        args = {"variable_id": str(var_id), "year": int(year)}
        if meta.get("subject_id"):
            args["subject_id"] = str(meta["subject_id"])
        if meta.get("unit"):
            args["unit"] = str(meta["unit"])
        result = {"variable": title, "unit": meta.get("unit", ""), "year": int(year),
                  "value": meta.get("value"), "measure": meta.get("measure", "")}
        return tool_name, args, result, (text or title), meta

    if source == "isap":
        pub, year, pos = meta.get("publisher"), meta.get("year"), meta.get("pos")
        if not pub or year is None or pos is None:
            return None
        args = {"publisher": str(pub), "year": int(year), "position": int(pos)}
        result = {"title": title, "text": truncate(text, 1500)}
        return tool_name, args, result, truncate(text, 800), meta

    if source == "dane.gov.pl":
        if not title:
            return None
        args = {"query": title}
        result = {"title": title, "notes": truncate(text, 1500),
                  "license": rec.get("license", "unknown")}
        return tool_name, args, result, truncate(text, 800), meta

    return None


def agentic_pairs(rec: dict, per_record: int) -> list[dict]:
    """Build validated tool-use trajectories grounded in the record's source API.

    All tools are offered on every sample so the model learns tool *selection*, not just
    argument filling. Samples whose arguments fail JSON-Schema validation are dropped.
    """
    derived = _agentic_call(rec)
    if derived is None:
        return []
    tool_name, args, result, final_answer, meta = derived
    source = rec.get("source", "")
    qs = _AGENTIC_QUESTIONS.get(source, [])[:per_record]
    fmt = {
        "title": (rec.get("title") or "ten zasób").strip(),
        "unit": meta.get("unit", ""),
        "year": meta.get("year", ""),
        "publisher": meta.get("publisher", ""),
        "pos": meta.get("pos", ""),
    }
    out = []
    for q in qs:
        sample = make_tool_sample(
            user=q.format(**fmt),
            tool_name=tool_name,
            arguments=args,
            tool_result=result,
            final_answer=final_answer,
            tools=ALL_TOOLS,
            source=source,
            license=rec.get("license", "unknown"),
            snapshot_date=rec.get("snapshot_date", ""),
        )
        ok, _ = validate_sample(sample)
        if ok:
            out.append(sample)
    return out


def llm_pairs(rec: dict, per_record: int) -> list[dict]:
    # TODO: wire to a teacher model to diversify questions / write fuller answers
    # grounded strictly in rec["text"]. Keep prompts + outputs logged for audit.
    # For tool-use data without an LLM, use --mode agentic (deterministic, grounded).
    raise NotImplementedError("LLM mode not wired yet — use --mode template or agentic.")


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
    ap.add_argument("--mode", choices=["template", "agentic", "llm"], default="template")
    ap.add_argument("--per-record", type=int, default=2)
    args = ap.parse_args()

    gen = {"template": template_pairs, "agentic": agentic_pairs,
           "llm": llm_pairs}[args.mode]
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
