"""Auto target-module detection, incl. fused-QKV architectures."""

import pytest

from train._common import _best_pattern, _resolve_target_modules


class FakeModel:
    """Minimal stand-in exposing named_modules() with given leaf names."""

    def __init__(self, leaf_names):
        self._leaves = leaf_names

    def named_modules(self):
        for i, n in enumerate(self._leaves):
            yield f"model.layers.{i}.{n}", object()


def test_best_pattern_picks_max_hits():
    names = {"qkv_proj", "o_proj"}
    # Fused pattern (2 hits) must beat the split pattern (only o_proj, 1 hit).
    assert set(_best_pattern([["q_proj", "k_proj", "v_proj", "o_proj"],
                              ["qkv_proj", "o_proj"]], names)) == {"qkv_proj", "o_proj"}


def test_llama_detection():
    m = FakeModel(["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"])
    got = set(_resolve_target_modules(m, "auto"))
    assert got == {"q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"}


def test_phi3_fused_detection():
    # Phi-3: fused qkv_proj + gate_up_proj. Old code would have detected only o_proj/down_proj.
    m = FakeModel(["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
    got = set(_resolve_target_modules(m, "auto"))
    assert "qkv_proj" in got and "gate_up_proj" in got
    assert got == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}


def test_ssm_included():
    m = FakeModel(["q_proj", "v_proj", "in_proj_qkv", "dt_proj", "gate_proj"])
    got = set(_resolve_target_modules(m, "auto"))
    assert {"in_proj_qkv", "dt_proj"} <= got


def test_explicit_list_passthrough():
    m = FakeModel(["whatever"])
    assert _resolve_target_modules(m, ["a", "b"]) == ["a", "b"]


def test_no_match_raises():
    m = FakeModel(["some_unknown_layer", "another"])
    with pytest.raises(ValueError):
        _resolve_target_modules(m, "auto")
