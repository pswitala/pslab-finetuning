"""Tool catalog registry + source mapping."""

import pytest

from common.tool_catalog import (
    ALL_TOOLS, get_tool, parameters_schema, tool_for_source,
)


def test_all_tools_wellformed():
    names = set()
    for t in ALL_TOOLS:
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "required" in fn["parameters"]
        names.add(fn["name"])
    assert names == {"gus_bdl_query", "dane_gov_search", "isap_lookup"}


def test_source_mapping():
    assert tool_for_source("gus_bdl") == "gus_bdl_query"
    assert tool_for_source("isap") == "isap_lookup"
    assert tool_for_source("dane.gov.pl") == "dane_gov_search"
    assert tool_for_source("nope") is None


def test_get_tool_and_schema():
    assert get_tool("isap_lookup")["function"]["name"] == "isap_lookup"
    assert parameters_schema("isap_lookup")["required"] == ["publisher", "year", "position"]
    with pytest.raises(KeyError):
        get_tool("does_not_exist")
