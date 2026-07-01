"""Helpers for building and validating agentic (tool-use) training samples.

A tool-use sample is a chat conversation that carries both the available `tools` and an
assistant turn that emits `tool_calls`, followed by the tool's result and a final grounded
answer — the OpenAI/Qwen function-calling shape that `tokenizer.apply_chat_template(...,
tools=...)` renders:

    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": "gus_bdl_query", "arguments": {...}}}]},
        {"role": "tool", "name": "gus_bdl_query", "tool_call_id": "call_0",
         "content": "<result json>"},
        {"role": "assistant", "content": "<final answer grounded in the tool result>"}],
     "tools": [ {full function schema}, ... ],
     "source": ..., "license": ..., "snapshot_date": ...}

`arguments` is stored as a dict (HF chat templates expect a dict, e.g. Qwen does
`tool_call.arguments | tojson`), and validated against the tool's parameters JSON Schema at
build time via `jsonschema` (already a dependency).
"""

from __future__ import annotations

import json
from typing import Any

# Local import so this module is importable without scripts/ on the path elsewhere.
try:
    from common.tool_catalog import get_tool
except ImportError:  # when imported as scripts.common.tooling
    from scripts.common.tool_catalog import get_tool  # type: ignore


def make_tool_sample(
    *,
    user: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_result: Any,
    final_answer: str,
    tools: list[dict],
    source: str = "",
    license: str = "unknown",
    snapshot_date: str = "",
    call_id: str = "call_0",
) -> dict:
    """Build a single-tool-call conversation sample (see module docstring)."""
    result_str = tool_result if isinstance(tool_result, str) else json.dumps(
        tool_result, ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "",
             "tool_calls": [{
                 "id": call_id,
                 "type": "function",
                 "function": {"name": tool_name, "arguments": arguments},
             }]},
            {"role": "tool", "name": tool_name, "tool_call_id": call_id,
             "content": result_str},
            {"role": "assistant", "content": final_answer},
        ],
        "tools": tools,
        "source": source,
        "license": license,
        "snapshot_date": snapshot_date,
    }


def validate_arguments(tool_name: str, arguments: dict) -> tuple[bool, str]:
    """Validate `arguments` against the tool's parameters JSON Schema.

    Returns (ok, error_message). ok=True means the arguments conform. Falls back to a
    minimal required-keys check if jsonschema is unavailable.
    """
    try:
        schema = get_tool(tool_name)["function"]["parameters"]
    except KeyError:
        return False, f"unknown tool: {tool_name}"

    try:
        import jsonschema
    except ImportError:
        required = schema.get("required", [])
        missing = [k for k in required if k not in arguments]
        return (not missing), (f"missing required: {missing}" if missing else "")

    try:
        jsonschema.validate(arguments, schema)
        return True, ""
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        return False, exc.message


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Validate a full tool-use sample: structure + every tool_call's arguments."""
    msgs = sample.get("messages")
    if not msgs:
        return False, "no messages"
    calls = [c for m in msgs if m.get("role") == "assistant"
             for c in (m.get("tool_calls") or [])]
    if not calls:
        return False, "no tool_calls in any assistant turn"
    for c in calls:
        fn = c.get("function", {})
        name, args = fn.get("name"), fn.get("arguments")
        if not isinstance(args, dict):
            return False, f"arguments for {name} is not an object"
        ok, err = validate_arguments(name, args)
        if not ok:
            return False, f"{name}: {err}"
    return True, ""
