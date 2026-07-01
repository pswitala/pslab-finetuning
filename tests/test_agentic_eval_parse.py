"""Tool-call parsing + argument matching in the agentic eval."""

import importlib

# scripts/eval isn't a conventional package name; import by path-safe module string.
agentic_eval = importlib.import_module("eval.agentic_eval")
parse_tool_calls = agentic_eval.parse_tool_calls
args_match = agentic_eval.args_match
_balanced = agentic_eval._balanced_json_objects


def test_balanced_handles_nesting():
    objs = _balanced('noise {"a": {"b": 1}} tail {"c": 2}')
    assert objs == ['{"a": {"b": 1}}', '{"c": 2}']


def test_parse_qwen_tool_call():
    text = '<tool_call>{"name": "gus_bdl_query", "arguments": {"variable_id": "1", "year": 2020}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "gus_bdl_query",
                      "arguments": {"variable_id": "1", "year": 2020}}]


def test_parse_function_wrapper():
    text = '{"function": {"name": "isap_lookup", "arguments": {"publisher": "DU", "year": 2020, "position": 5}}}'
    calls = parse_tool_calls(text)
    assert calls[0]["name"] == "isap_lookup"
    assert calls[0]["arguments"]["position"] == 5


def test_parse_stringified_arguments():
    text = '{"name": "dane_gov_search", "arguments": "{\\"query\\": \\"budżet\\"}"}'
    calls = parse_tool_calls(text)
    assert calls[0]["arguments"] == {"query": "budżet"}


def test_args_match_normalizes_types():
    # gold year is int, prediction may be a string — still a match.
    assert args_match({"variable_id": "1", "year": "2020"}, {"year": 2020})
    assert not args_match({"year": "2019"}, {"year": 2020})


def test_no_tool_call():
    assert parse_tool_calls("nie wiem, przepraszam") == []
