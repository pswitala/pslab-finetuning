"""Multi-GPU trainer-arg derivation: batch policy, DDP knobs, device placement.

Pure config arithmetic — no GPU, no torch model, no launcher. Guards the one piece of the
DDP wiring that is easy to get silently wrong: whether a 4-GPU run reproduces the
single-GPU effective batch or quadruples it.
"""

import sys
import types

import pytest

from train._common import (
    _is_vision_param,
    _resolve_device_map,
    dist_env,
    distributed_training_args,
    is_main_process,
)


def launch(monkeypatch, *, rank=0, local_rank=0, world_size=1):
    """Fake a torchrun environment."""
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


@pytest.fixture
def set_device_calls(monkeypatch):
    """Stub `import torch` inside _resolve_device_map; returns the set_device call log.

    _resolve_device_map is pure placement logic, and the test env (and CI, which installs
    only ruff/pytest/pyyaml) has no torch — so fake the one attribute it touches.
    """
    calls: list[int] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(set_device=calls.append)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls


def cfg(grad_accum=8, per_device=2, **distributed):
    out = {"per_device_train_batch_size": per_device,
           "gradient_accumulation_steps": grad_accum}
    if distributed:
        out["distributed"] = distributed
    return out


# --- environment detection ---------------------------------------------------

def test_unlaunched_env_is_single_process(monkeypatch):
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)
    assert dist_env() == (0, 0, 1)
    assert is_main_process()


def test_non_zero_rank_is_not_main(monkeypatch):
    launch(monkeypatch, rank=2, local_rank=2, world_size=4)
    assert dist_env() == (2, 2, 4)
    assert not is_main_process()


# --- effective batch policy --------------------------------------------------

def test_single_gpu_leaves_grad_accum_alone(monkeypatch):
    launch(monkeypatch)
    assert distributed_training_args(cfg(grad_accum=8), 4)["gradient_accumulation_steps"] == 8


def test_constant_policy_divides_by_world_size(monkeypatch):
    launch(monkeypatch, world_size=4)
    args = distributed_training_args(cfg(grad_accum=8, per_device=2), 4)
    assert args["gradient_accumulation_steps"] == 2
    # effective batch (2 per-device x 2 accum x 4 gpu) == the single-GPU 2 x 8 x 1
    assert 2 * args["gradient_accumulation_steps"] * 4 == 2 * 8 * 1


def test_constant_is_the_default_policy(monkeypatch):
    launch(monkeypatch, world_size=4)
    # no `distributed:` block at all
    assert distributed_training_args(cfg(grad_accum=8), 4)["gradient_accumulation_steps"] == 2


def test_scale_policy_keeps_grad_accum(monkeypatch):
    launch(monkeypatch, world_size=4)
    args = distributed_training_args(cfg(grad_accum=8, effective_batch="scale"), 4)
    assert args["gradient_accumulation_steps"] == 8


def test_indivisible_grad_accum_is_kept_not_rounded(monkeypatch):
    """3 accum steps over 4 ranks must not silently become 0 (which would break the run)."""
    launch(monkeypatch, world_size=4)
    args = distributed_training_args(cfg(grad_accum=3), 4)
    assert args["gradient_accumulation_steps"] == 3


def test_default_grad_accum_used_when_config_omits_it(monkeypatch):
    launch(monkeypatch, world_size=2)
    assert distributed_training_args({}, 8)["gradient_accumulation_steps"] == 4


# --- DDP knobs ---------------------------------------------------------------

def test_ddp_defaults(monkeypatch):
    launch(monkeypatch, world_size=4)
    args = distributed_training_args(cfg(), 4)
    assert args["ddp_find_unused_parameters"] is False
    assert args["dataloader_num_workers"] == 4
    assert args["ddp_timeout"] == 5400


def test_ddp_knobs_are_overridable(monkeypatch):
    launch(monkeypatch, world_size=4)
    args = distributed_training_args(
        cfg(ddp_find_unused_parameters=True, dataloader_num_workers=8, ddp_timeout=900), 4)
    assert args["ddp_find_unused_parameters"] is True
    assert args["dataloader_num_workers"] == 8
    assert args["ddp_timeout"] == 900


def test_dataloader_workers_default_to_zero_single_process(monkeypatch):
    launch(monkeypatch)
    assert distributed_training_args(cfg(), 4)["dataloader_num_workers"] == 0


# --- device placement --------------------------------------------------------

def test_device_map_is_auto_single_process(monkeypatch, set_device_calls):
    launch(monkeypatch)
    assert _resolve_device_map({}) == "auto"
    assert set_device_calls == []


def test_device_map_override_single_process(monkeypatch, set_device_calls):
    launch(monkeypatch)
    assert _resolve_device_map({"distributed": {"device_map": {"": 0}}}) == {"": 0}


def test_device_map_pins_to_local_rank_under_launcher(monkeypatch, set_device_calls):
    """The DDP-critical case: "auto" would shard each rank's replica across all GPUs."""
    launch(monkeypatch, rank=3, local_rank=3, world_size=4)
    # config still says "auto" — the launcher must win
    assert _resolve_device_map({"distributed": {"device_map": "auto"}}) == {"": 3}
    assert set_device_calls == [3]


# --- vision-tower detection (unused-parameter guard) -------------------------

@pytest.mark.parametrize("name", [
    "visual.blocks.0.mlp.gate_proj.weight",
    "vision_tower.encoder.layers.1.self_attn.q_proj.weight",
    # after PEFT rewraps the model, names gain a base_model.model. prefix
    "base_model.model.visual.blocks.0.mlp.gate_proj.lora_A.default.weight",
    "base_model.model.vision_encoder.layer.2.out_proj.lora_B.default.weight",
])
def test_vision_params_detected(name):
    assert _is_vision_param(name)


@pytest.mark.parametrize("name", [
    "model.layers.0.self_attn.q_proj.weight",
    "base_model.model.model.layers.5.mlp.down_proj.lora_A.default.weight",
    "lm_head.weight",
    # substring of a vision prefix must not match — component matching, not `in`
    "model.layers.0.visualizer_proj.weight",
])
def test_language_params_not_flagged_as_vision(name):
    assert not _is_vision_param(name)
