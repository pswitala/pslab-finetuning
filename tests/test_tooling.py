"""Tool-use sample construction + JSON-Schema validation."""

from common.tool_catalog import get_tool
from common.tooling import make_tool_sample, validate_arguments, validate_sample


def _sample(args):
    return make_tool_sample(
        user="Ile wyniósł wskaźnik X w 2020?",
        tool_name="gus_bdl_query",
        arguments=args,
        tool_result={"value": 42},
        final_answer="Wyniósł 42.",
        tools=[get_tool("gus_bdl_query")],
        source="gus_bdl",
    )


def test_sample_structure():
    s = _sample({"variable_id": "123", "year": 2020})
    roles = [m["role"] for m in s["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    call = s["messages"][1]["tool_calls"][0]
    assert call["function"]["name"] == "gus_bdl_query"
    assert isinstance(call["function"]["arguments"], dict)  # dict, not JSON string
    assert s["messages"][2]["tool_call_id"] == call["id"]
    assert s["tools"]


def test_valid_sample_passes():
    ok, err = validate_sample(_sample({"variable_id": "123", "year": 2020}))
    assert ok, err


def test_missing_required_fails():
    ok, _ = validate_sample(_sample({"year": 2020}))  # no variable_id
    assert not ok


def test_wrong_type_fails():
    ok, _ = validate_arguments("gus_bdl_query", {"variable_id": "1", "year": "2020"})
    assert not ok  # year must be integer


def test_unknown_tool_fails():
    ok, _ = validate_arguments("mystery", {})
    assert not ok
