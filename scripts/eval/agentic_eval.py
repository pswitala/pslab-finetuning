#!/usr/bin/env python3
"""Agentic (tool-use) eval — does the model emit the RIGHT tool call?

Complements catalog_eval.py (closed-book recall) by measuring function-calling ability:
given a Polish question and the available `tools`, does the model select the correct tool
and produce well-formed, correct arguments? This directly measures the agentic goal, and
is far more meaningful than token overlap.

Build a held-out set with the agentic builder (each line carries `messages`+`tools`, where
the gold assistant turn contains the reference tool_call):
    python scripts/process/build_sft_qa.py \
        --input "data/catalogs/_holdout/**/*.jsonl" \
        --out eval/data/agentic_holdout.jsonl --mode agentic --per-record 1

Then evaluate (hf | vllm | gguf, same as catalog_eval):
    python scripts/eval/agentic_eval.py --model models/dpo/merged \
        --qa eval/data/agentic_holdout.jsonl --out eval/results/agentic

Metrics:
  format_rate          fraction where a parseable tool_call was emitted
  tool_selection_acc   fraction where the tool NAME matches the gold call
  args_exact_acc       fraction where every gold argument is present with the right value
  schema_valid_rate    fraction where the emitted arguments validate against the schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tooling import validate_arguments  # noqa: E402


# --- gold-set loading ---------------------------------------------------------

def load_agentic(path: str) -> list[dict]:
    """Extract (question, tools, gold tool name+arguments, gold final answer)."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            tools = ex.get("tools", [])
            question = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            gold_call = next(
                (c for m in msgs if m.get("role") == "assistant"
                 for c in (m.get("tool_calls") or [])), None)
            if question is None or gold_call is None:
                continue
            fn = gold_call.get("function", {})
            items.append({
                "q": question,
                "tools": tools,
                "gold_name": fn.get("name"),
                "gold_args": fn.get("arguments") or {},
            })
    return items


# --- tool-call parsing --------------------------------------------------------

def _balanced_json_objects(text: str) -> list[str]:
    """Return top-level balanced {...} substrings (handles nested arguments)."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i + 1])
                start = None
    return objs


def _try_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_tool_calls(text: str) -> list[dict]:
    """Best-effort parse of tool calls from free-text model output.

    Handles Qwen-style `<tool_call>{...}</tool_call>` and bare JSON objects that carry a
    name + arguments (possibly nested under `function`). Returns [{name, arguments}].
    """
    calls = []
    for blob in _balanced_json_objects(text):
        obj = _try_json(blob)
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            args = _try_json(args) or {}
        if name and isinstance(args, dict):
            calls.append({"name": name, "arguments": args})
    return calls


# --- scoring ------------------------------------------------------------------

def _norm(v) -> str:
    return str(v).strip().lower()


def args_match(pred: dict, gold: dict) -> bool:
    """True if every gold argument appears in pred with an equal (normalized) value."""
    return all(k in pred and _norm(pred[k]) == _norm(v) for k, v in gold.items())


def score(pred_calls: list[dict], gold_name: str, gold_args: dict) -> dict:
    pred = pred_calls[0] if pred_calls else None
    if pred is None:
        return {"format": 0, "name": 0, "args": 0, "schema": 0}
    schema_ok, _ = validate_arguments(pred["name"], pred["arguments"])
    name_ok = pred["name"] == gold_name
    return {
        "format": 1,
        "name": int(name_ok),
        "args": int(name_ok and args_match(pred["arguments"], gold_args)),
        "schema": int(schema_ok),
    }


# --- inference backends (mirror catalog_eval.py) ------------------------------

def _prompt(tok, question: str, tools: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "tools": tools}
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": question}], enable_thinking=False, **kwargs)
    except TypeError:
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": question}], **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return tok.apply_chat_template(
                [{"role": "user", "content": question}], **kwargs)


def _infer_hf(args, qa: list[dict]) -> list[list[dict]]:
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

    preds = []
    for ex in qa:
        prompt = _prompt(tok, ex["q"], ex["tools"])
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        out = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        preds.append(parse_tool_calls(out))
    return preds


def _infer_vllm(args, qa: list[dict]) -> list[list[dict]]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=4096)
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    prompts = [_prompt(tok, ex["q"], ex["tools"]) for ex in qa]
    outputs = llm.generate(prompts, params)
    return [parse_tool_calls(o.outputs[0].text) for o in outputs]


def _infer_gguf(args, qa: list[dict]) -> list[list[dict]]:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            f"[agentic_eval] llama-cpp-python not available: {exc}\n"
            "Install with GPU support: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
        ) from exc

    llm = Llama(model_path=args.model, n_ctx=4096, n_gpu_layers=-1, verbose=False)
    preds = []
    for ex in qa:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": ex["q"]}],
            tools=ex["tools"], max_tokens=args.max_new_tokens, temperature=0.0)
        msg = out["choices"][0]["message"]
        calls = []
        for c in (msg.get("tool_calls") or []):
            fn = c.get("function", {})
            a = fn.get("arguments")
            calls.append({"name": fn.get("name"),
                          "arguments": _try_json(a) if isinstance(a, str) else (a or {})})
        # Some GGUF chat handlers emit the call in content instead of tool_calls.
        if not calls and msg.get("content"):
            calls = parse_tool_calls(msg["content"])
        preds.append(calls)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model path/id, or .gguf file path when --backend gguf")
    ap.add_argument("--qa", required=True, help="agentic holdout jsonl (mode=agentic)")
    ap.add_argument("--out", default="eval/results/agentic")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--backend", choices=["hf", "vllm", "gguf"], default="hf")
    args = ap.parse_args()

    qa = load_agentic(args.qa)
    if not qa:
        print("[agentic_eval] no tool-use items found; build with --mode agentic")
        return 1

    infer = {"hf": _infer_hf, "vllm": _infer_vllm, "gguf": _infer_gguf}[args.backend]
    preds = infer(args, qa)

    agg = {"format": 0, "name": 0, "args": 0, "schema": 0}
    details = []
    for ex, pred_calls in zip(qa, preds):
        s = score(pred_calls, ex["gold_name"], ex["gold_args"])
        for k in agg:
            agg[k] += s[k]
        details.append({
            "q": ex["q"], "gold_name": ex["gold_name"], "gold_args": ex["gold_args"],
            "pred": pred_calls[0] if pred_calls else None, **s,
        })

    n = len(qa)
    summary = {
        "n": n,
        "format_rate": agg["format"] / n,
        "tool_selection_acc": agg["name"] / n,
        "args_exact_acc": agg["args"] / n,
        "schema_valid_rate": agg["schema"] / n,
        "model": args.model,
        "backend": args.backend,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in details))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
