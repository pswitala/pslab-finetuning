"""Shared helpers for the QLoRA training scripts (CPT / SFT / DPO).

Supports any HuggingFace causal-LM or VLM via PEFT. Tries Unsloth first;
falls back to PEFT + bitsandbytes automatically.

Config keys consumed here:
  base_model            — HF model ID or local path
  lora.target_modules   — projection name list, or "auto" for auto-detection
  freeze_vision_encoder — "auto" | true | false  (default: "auto")
  load_in_4bit          — NF4 quantization
  use_unsloth           — try the Unsloth fast path first (default: true). Set false
                          (or export PSLAB_NO_UNSLOTH=1) to go straight to PEFT+bnb.
                          Needed for Qwen3_5/Qwen3.6 VLMs, which Unsloth loads in bf16
                          instead of 4-bit — fine on 96 GB, OOMs on <=48 GB cards.
  bf16, gradient_checkpointing
  distributed.*          — multi-GPU policy (effective_batch, ddp_find_unused_parameters,
                           dataloader_num_workers, ddp_timeout, device_map). Only consulted
                           under torchrun / accelerate launch; see distributed_training_args.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- distributed (multi-GPU) helpers -----------------------------------------
# The scripts run unchanged as `python scripts/train/cpt.py` (single process) or under
# `torchrun --nproc_per_node=N` / `accelerate launch --num_processes N`, which both export
# RANK / LOCAL_RANK / WORLD_SIZE. Everything below keys off those env vars, so nothing
# distributed-specific happens unless a launcher is actually used.

def dist_env() -> tuple[int, int, int]:
    """(rank, local_rank, world_size) from the launcher env; (0, 0, 1) if unlaunched."""
    return (int(os.environ.get("RANK", 0)),
            int(os.environ.get("LOCAL_RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)))


def is_main_process() -> bool:
    return dist_env()[0] == 0


def rank0_print(*args: Any, **kwargs: Any) -> None:
    """print() on rank 0 only — keeps a 4-GPU run's log readable."""
    if is_main_process():
        print(*args, **kwargs)


def _resolve_device_map(cfg: dict) -> Any:
    """Model placement for from_pretrained.

    Under a launcher each rank holds a FULL replica pinned to its own GPU ({"": local_rank});
    DDP then all-reduces only the LoRA gradients (tens of MB), which PCIe handles easily —
    the RTX 6000 Pro Blackwell has no NVLink, so keeping inter-GPU traffic small matters.

    device_map="auto" must NOT be used here: with no per-rank CUDA_VISIBLE_DEVICES every rank
    sees every GPU and independently pipeline-shards its own copy across all of them, which
    breaks DDP (parameters end up off the rank's device). TRL forces device_map=None inside
    SFTTrainer for the same reason, but that guard only fires when TRL loads the model itself.

    Single-process runs keep "auto", so one model still spreads across the visible cards.
    """
    import torch
    _, local_rank, world = dist_env()
    if world > 1:
        torch.cuda.set_device(local_rank)
        return {"": local_rank}
    return (cfg.get("distributed") or {}).get("device_map", "auto")


def distributed_training_args(cfg: dict, default_grad_accum: int) -> dict[str, Any]:
    """Trainer kwargs that depend on world size — splatted into SFTConfig/DPOConfig.

    Centralizes the multi-GPU policy so cpt/sft/dpo stay identical. Config keys live under
    a `distributed:` block; all are optional and absent keys reproduce single-GPU behavior.

    effective_batch (per_device x grad_accum x world_size):
      "constant" (default) — divide grad_accum by world_size so the effective batch matches
                             the single-GPU run exactly. Same optimization math, ~Nx faster
                             wall clock, no learning-rate retuning.
      "scale"              — use grad_accum as-is; effective batch grows with world_size.
                             Fewer optimizer steps over the same data; retune lr/warmup.
    """
    d = cfg.get("distributed") or {}
    _, _, world = dist_env()
    per_device = int(cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", default_grad_accum))

    policy = d.get("effective_batch", "constant")
    if world > 1 and policy == "constant":
        if grad_accum % world == 0:
            grad_accum //= world
        else:
            rank0_print(
                f"[_common] gradient_accumulation_steps={grad_accum} is not divisible by "
                f"world_size={world}; keeping it as-is. The effective batch is {world}x the "
                "single-GPU value — pick a divisible value or set "
                "distributed.effective_batch: scale to silence this.")
    if world > 1:
        rank0_print(
            f"[_common] distributed: world_size={world} policy={policy} "
            f"effective_batch={per_device * grad_accum * world} "
            f"({per_device} per-device x {grad_accum} accum x {world} gpu)")

    return dict(
        gradient_accumulation_steps=grad_accum,
        # LoRA on a frozen base leaves no unused parameters once vision-tower adapters are
        # frozen (see _maybe_freeze_vision_encoder); False avoids DDP's per-step param sweep.
        ddp_find_unused_parameters=d.get("ddp_find_unused_parameters", False),
        dataloader_num_workers=d.get("dataloader_num_workers", 4 if world > 1 else 0),
        # A cold 27B NF4 load can outlast the 1800 s default and trip an NCCL timeout.
        ddp_timeout=d.get("ddp_timeout", 5400),
    )


def wandb_report_to() -> list[str]:
    """['wandb'] if WANDB_API_KEY is set, else [] — for SFTConfig/DPOConfig report_to.

    Shared by cpt/sft/dpo so experiment tracking is enabled consistently across stages.
    """
    import os
    return ["wandb"] if os.environ.get("WANDB_API_KEY") else []


def _apply_template(tokenizer, messages, enable_thinking: bool,
                    add_generation_prompt: bool, tools=None) -> str:
    """apply_chat_template that degrades gracefully across tokenizer capabilities.

    Progressively drops kwargs the tokenizer doesn't accept: enable_thinking first
    (older tokenizers), then tools (non-tool-capable templates). Shared by SFT and DPO.
    """
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)   # template predates tool support
            return tokenizer.apply_chat_template(messages, **kwargs)


def render_prompt_completion(tokenizer, messages, tools,
                             enable_thinking: bool) -> tuple[str, str] | None:
    """Split a conversation into (prompt_text, completion_text) for SFT.

    Enables TRL's `completion_only_loss` (loss computed on the completion only): the prompt
    is everything up to the first assistant turn (rendered with add_generation_prompt=True,
    plus any `tools`); the completion is the remainder (rendered full, prompt prefix sliced
    off — the same conversion TRL uses internally). enable_thinking=False suppresses <think>.

    For multi-turn tool-use (user → assistant tool_call → tool → assistant answer) the whole
    assistant-side trajectory is the completion; loss over the injected tool-result tokens is
    a minor, tolerable imperfection versus the alternative of computing loss on the user turn.

    Returns None if there is no assistant turn to learn from.
    """
    first_asst = next((i for i, m in enumerate(messages)
                       if m.get("role") == "assistant"), None)
    if first_asst is None:
        return None
    prompt_msgs = messages[:first_asst]
    prompt_text = _apply_template(tokenizer, prompt_msgs, enable_thinking,
                                  add_generation_prompt=True, tools=tools)
    full_text = _apply_template(tokenizer, messages, enable_thinking,
                                add_generation_prompt=False, tools=tools)
    completion = (full_text[len(prompt_text):]
                  if full_text.startswith(prompt_text) else full_text)
    return prompt_text, completion


def render_preference_to_text(tokenizer, ex: dict, enable_thinking: bool = False) -> dict:
    """Convert a conversational DPO example to TRL 'standard' string format.

    DPOTrainer applies the tokenizer's chat template to conversational
    prompt/chosen/rejected using its DEFAULT kwargs, so enable_thinking is never honored
    and <think> blocks are not suppressed. Rendering here (with enable_thinking) fixes that.

    Mirrors TRL's own conversational→standard conversion: the completion is the full
    prompt+turns render with the prompt prefix sliced off. Flat-string examples (prompt is
    already a str) are returned unchanged.
    """
    prompt = ex.get("prompt")
    if not isinstance(prompt, list):
        return ex  # already standard (flat strings) — nothing to render
    prompt_text = _apply_template(tokenizer, prompt, enable_thinking,
                                  add_generation_prompt=True)

    def _completion(turns) -> str:
        turns = turns if isinstance(turns, list) else [turns]
        full = _apply_template(tokenizer, prompt + turns, enable_thinking,
                               add_generation_prompt=False)
        # Well-behaved (ChatML/Qwen) templates render prompt+assistant so that the prefix
        # equals the add_generation_prompt render; slice it off. Fall back to the full
        # render if the prefix doesn't match (unusual template).
        return full[len(prompt_text):] if full.startswith(prompt_text) else full

    out = dict(ex)
    out["prompt"] = prompt_text
    out["chosen"] = _completion(ex.get("chosen"))
    out["rejected"] = _completion(ex.get("rejected"))
    return out


def make_training_callbacks(cfg: dict) -> list:
    """Trainer callbacks that abort a diverging run early (shared by cpt/sft/dpo).

    A too-high LR (or a bad batch) shows up as exploding grad_norm and rising/NaN loss. Left
    alone, a multi-day run keeps burning hours going nowhere. StabilityGuard watches the
    per-step training log and calls `control.should_training_stop` on NaN loss or after
    `patience` consecutive grad_norm spikes — so it stops in a handful of logging steps, and
    the periodic checkpoints (save_steps) remain your rollback points to the best step.

    Steered by config keys, each overridable by an env var (env wins) so a run can be tuned
    without editing YAML:
        stability_guard: true|false      env PSLAB_STABILITY_GUARD = 0/1
        max_grad_norm_abort: 100.0       env PSLAB_MAX_GRAD_NORM_ABORT   (pre-clip norm)
        stability_patience: 3            env PSLAB_STABILITY_PATIENCE    (consecutive spikes)
    """
    import os
    from transformers import TrainerCallback

    enabled = bool(cfg.get("stability_guard", True))
    flag = os.environ.get("PSLAB_STABILITY_GUARD")
    if flag in ("0", "false", "False"):
        enabled = False
    elif flag in ("1", "true", "True"):
        enabled = True
    if not enabled:
        rank0_print("[_common] stability guard disabled")
        return []

    max_gn = float(os.environ.get("PSLAB_MAX_GRAD_NORM_ABORT")
                   or cfg.get("max_grad_norm_abort", 100.0))
    patience = int(os.environ.get("PSLAB_STABILITY_PATIENCE")
                   or cfg.get("stability_patience", 3))

    class StabilityGuard(TrainerCallback):
        def __init__(self, max_grad_norm: float, patience: int):
            self.max_grad_norm = max_grad_norm
            self.patience = patience
            self.bad = 0

        @staticmethod
        def _num(v):
            try:
                return float(v)            # logs may carry floats or preformatted strings
            except (TypeError, ValueError):
                return None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = self._num(logs.get("loss"))
            grad_norm = self._num(logs.get("grad_norm"))
            if loss is not None and loss != loss:  # NaN
                rank0_print(f"[stability-guard] NaN loss at step {state.global_step} — stopping. "
                      "Lower learning_rate and resume from the last good checkpoint.")
                control.should_training_stop = True
                return
            if grad_norm is not None and grad_norm > self.max_grad_norm:
                self.bad += 1
                rank0_print(f"[stability-guard] grad_norm {grad_norm:.1f} > {self.max_grad_norm} "
                      f"at step {state.global_step} ({self.bad}/{self.patience} spikes)")
                if self.bad >= self.patience:
                    rank0_print("[stability-guard] repeated grad-norm spikes — stopping. Lower "
                          "learning_rate (or raise warmup) and resume from the best checkpoint.")
                    control.should_training_stop = True
            else:
                # Decay (not hard-reset): a lone transient spike is forgiven, but intermittent
                # spikes that keep recurring — as in a diverging warmup ramp — still accumulate.
                self.bad = max(0, self.bad - 1)

    rank0_print(f"[_common] stability guard on: abort on NaN loss or {patience} consecutive "
          f"grad_norm spikes > {max_gn}")
    return [StabilityGuard(max_gn, patience)]


# Attention projection patterns. Fused-QKV variants (Phi-3, MPT, Falcon, GPT-2) are
# listed alongside the split variant; _resolve_target_modules picks the pattern with the
# MOST matches, so an arch that has both `o_proj` and `qkv_proj` (Phi-3) correctly
# selects the fused pattern instead of just `o_proj`.
_ATTN_PATTERNS = [
    ["q_proj", "k_proj", "v_proj", "o_proj"],   # Llama, Mistral, Qwen attn, Gemma
    ["qkv_proj", "o_proj"],                        # Phi-3 (fused QKV)
    ["Wqkv", "out_proj"],                         # MPT / some fused-QKV archs
    ["c_attn", "c_proj"],                         # GPT-2 / GPT-NeoX (fused QKV)
    ["to_q", "to_k", "to_v", "to_out"],           # Diffusers / DiT
    ["query", "key", "value", "dense"],            # BERT / RoBERTa
    ["query_key_value", "dense"],                  # Falcon / older MPT
]
_MLP_PATTERNS = [
    ["gate_proj", "up_proj", "down_proj"],         # Llama / Mistral / Qwen MLP
    ["gate_up_proj", "down_proj"],                 # Phi-3 (fused gate/up)
    ["fc1", "fc2"],                                # GPT-2 / BERT MLP
    ["w1", "w2", "w3"],                            # some custom architectures
]
_SSM_CANDIDATES = [
    "in_proj_qkv", "out_proj", "in_proj_z", "in_proj_a", "in_proj_b",
    "in_proj", "x_proj", "dt_proj",               # Mamba / Qwen SSM
]


def _best_pattern(patterns: list[list[str]], names: set[str]) -> list[str]:
    """Return the pattern with the most leaf-name matches (ties -> earliest listed)."""
    best: list[str] = []
    best_hits = 0
    for pat in patterns:
        hits = [p for p in pat if p in names]
        if len(hits) > best_hits:
            best, best_hits = hits, len(hits)
    return best


def _resolve_target_modules(model: Any, cfg_modules: Any) -> list[str] | Any:
    """Return resolved LoRA target_modules list.

    "auto" → inspect model.named_modules() and pick the best-matching attention + MLP
             projection pattern (by number of matches), plus any SSM projections.
    list   → returned unchanged (user override).

    Raises ValueError if "auto" detects nothing — silently targeting nonexistent
    modules (the old ["q_proj","v_proj"] fallback) produces a broken/no-op adapter on
    fused-QKV architectures, so we fail loudly with guidance instead.
    """
    if cfg_modules != "auto":
        return cfg_modules
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    detected: list[str] = []
    detected.extend(_best_pattern(_ATTN_PATTERNS, names))
    detected.extend(_best_pattern(_MLP_PATTERNS, names))
    detected.extend(p for p in _SSM_CANDIDATES if p in names)
    seen: set[str] = set()
    unique: list[str] = []
    for x in detected:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    if unique:
        rank0_print(f"[_common] auto-detected target_modules: {unique}")
        return unique
    sample = sorted(n for n in names if "proj" in n or "attn" in n or "fc" in n)
    raise ValueError(
        "target_modules: auto detected no known projection modules for this "
        "architecture. Set lora.target_modules explicitly in the config "
        "(see configs/models/ for presets). Candidate leaf module names found: "
        f"{sample or sorted(names)[:40]}"
    )


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    backend: str  # "unsloth" | "peft"


def load_model_and_tokenizer(cfg: dict) -> LoadedModel:
    """Load model + attach QLoRA adapter.

    Tries Unsloth first (fast kernels); falls back to PEFT + bitsandbytes
    if Unsloth does not support this architecture.

    Set use_unsloth: false (or PSLAB_NO_UNSLOTH=1) to skip Unsloth entirely. Required
    for Qwen3_5/Qwen3.6 VLMs: Unsloth "succeeds" but loads them in bf16 (ignoring
    load_in_4bit), so no exception fires to trigger the fallback — and a 27B bf16 model
    OOMs on a 48 GB card. The explicit PEFT+bnb path below always applies NF4.
    """
    import os
    if not cfg.get("use_unsloth", True) or os.environ.get("PSLAB_NO_UNSLOTH"):
        rank0_print("[_common] Unsloth disabled (use_unsloth=false / PSLAB_NO_UNSLOTH); "
              "loading via PEFT + bitsandbytes.")
        return _load_peft_fallback(cfg)
    try:
        from unsloth import FastLanguageModel
        import torch
        lora = cfg["lora"]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["base_model"],
            max_seq_length=cfg.get("max_seq_len", 4096),
            load_in_4bit=cfg.get("load_in_4bit", True),
            dtype=torch.bfloat16 if cfg.get("bf16", True) else None,
        )
        target_modules = _resolve_target_modules(model, lora["target_modules"])
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora.get("dropout", 0.0),
            target_modules=target_modules,
            use_rslora=lora.get("use_rslora", False),
            use_dora=lora.get("use_dora", False),
            use_gradient_checkpointing="unsloth" if cfg.get("gradient_checkpointing", True) else False,
            random_state=cfg.get("seed", 3407),
        )
        _maybe_freeze_vision_encoder(model, cfg)
        return LoadedModel(model, tokenizer, "unsloth")
    except Exception as exc:  # noqa: BLE001
        # OOM won't be fixed by the PEFT fallback — surface it instead of masking it.
        import torch
        if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)):
            raise
        # Anything else (Unsloth not installed / unsupported arch / bad kwarg): log the
        # full reason loudly, then fall back to PEFT. Silent fallback hid real bugs.
        import traceback
        rank0_print("[_common] " + "=" * 68)
        rank0_print(f"[_common] Unsloth path failed ({exc.__class__.__name__}: {exc}); "
              "falling back to PEFT + bitsandbytes.")
        traceback.print_exc()
        rank0_print("[_common] " + "=" * 68)
        return _load_peft_fallback(cfg)


# Ordered by likelihood for our targets (decoder-only LMs and text-generating VLMs).
# transformers v5 renamed the VLM class AutoModelForVision2Seq -> AutoModelForImageTextToText;
# the old name is kept last for pre-v5 compatibility. Classes absent in the installed
# transformers raise AttributeError and are skipped.
_MODEL_LOADERS = [
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForSeq2SeqLM",
]


def _is_attn_impl_error(exc: Any) -> bool:
    """True if an exception looks like an unsupported `attn_implementation`.

    Lets the loader retry with eager attention only when SDPA is genuinely unsupported by
    the architecture, rather than masking unrelated load failures (wrong arch, OOM, etc.).
    """
    if exc is None:
        return False
    text = str(exc).lower()
    return (
        "attn_implementation" in text
        or "scaled_dot_product" in text
        or ("does not support" in text and "attention" in text)
    )


def _torch_dtype(cfg: dict):
    """Compute dtype for the PEFT path from config, honoring bf16: false.

    bf16: true (default) -> bfloat16. bf16: false -> float16 (fp16 is the practical
    half-precision alternative on GPU; the bnb 4-bit compute dtype is set separately).
    """
    import torch
    return torch.bfloat16 if cfg.get("bf16", True) else torch.float16


def _disable_broken_causal_conv1d() -> None:
    """Neutralize an ABI-mismatched causal_conv1d wheel before loading SSM models.

    Qwen3.6/Qwen3_5 Gated DeltaNet layers import the causal_conv1d CUDA extension. If
    the installed wheel was built against a different torch ABI, importing it raises
    ImportError ("undefined symbol: ...c10_cuda_check_implementation..."), which aborts
    model loading in the PEFT path. When the extension is broken we tell transformers
    the package is unavailable so the model uses the (slower) torch SSM fallback — the
    same workaround Unsloth applies ("Detected broken causal_conv1d binary").

    A healthy install imports cleanly and is left untouched, so the fast path stays
    enabled wherever it actually works (e.g. the 96 GB reference rig).
    """
    try:
        import causal_conv1d_cuda  # noqa: F401  # the compiled extension that may be broken
        return  # healthy — keep the fast path
    except ImportError as exc:
        try:
            import transformers.utils.import_utils as iu
            iu.is_causal_conv1d_available = lambda *a, **k: False
        except Exception:  # noqa: BLE001
            return
        rank0_print(f"[_common] broken/absent causal_conv1d ({exc.__class__.__name__}); "
              "forcing torch SSM fallback")
    except Exception:  # noqa: BLE001
        return


def _load_peft_fallback(cfg: dict) -> LoadedModel:
    """PEFT path — used when Unsloth is unavailable or does not support this architecture."""
    import importlib
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    _disable_broken_causal_conv1d()

    lora = cfg["lora"]
    quant = None
    if cfg.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)

    load_kwargs = dict(
        quantization_config=quant,
        torch_dtype=_torch_dtype(cfg),   # honor bf16: false (was hardcoded bfloat16)
        device_map=_resolve_device_map(cfg),   # {"": local_rank} under a launcher, else auto
        trust_remote_code=True,
    )

    # SDPA attention: PyTorch-native, Blackwell-friendly, and far faster than eager at long
    # sequence lengths — many architectures otherwise default to eager (~O(len²) Python-heavy)
    # and crawl. flash-attn is intentionally avoided on Blackwell. Overridable via
    # cfg["attn_implementation"]; falls back to eager for architectures that lack SDPA.
    attn_pref = cfg.get("attn_implementation", "sdpa")

    def _try_loaders(attn_impl):
        exc_seen = None
        for cls_name in _MODEL_LOADERS:
            try:
                cls = getattr(importlib.import_module("transformers"), cls_name)
                m = cls.from_pretrained(
                    cfg["base_model"], attn_implementation=attn_impl, **load_kwargs)
                rank0_print(f"[_common] loaded with {cls_name} (attn_implementation={attn_impl})")
                return m, cls_name, None
            except Exception as exc:  # noqa: BLE001
                exc_seen = exc
                msg = str(exc).splitlines()[0] if str(exc) else ""
                rank0_print(f"[_common] {cls_name} failed (attn={attn_impl}) "
                      f"({exc.__class__.__name__}: {msg}); trying next...")
        return None, None, exc_seen

    model, used_cls_name, last_exc = _try_loaders(attn_pref)
    if model is None and attn_pref != "eager" and _is_attn_impl_error(last_exc):
        rank0_print(f"[_common] attn_implementation={attn_pref} unsupported here; "
                    "retrying with eager")
        model, used_cls_name, last_exc = _try_loaders("eager")
    if model is None:
        raise RuntimeError(
            f"Could not load {cfg['base_model']} with any Auto class. "
            f"Last error: {last_exc}"
        ) from last_exc

    _maybe_freeze_vision_encoder(model, cfg)

    grad_ckpt = cfg.get("gradient_checkpointing", True)
    # use_reentrant=False is required for multi-GPU QLoRA: the reentrant autograd path
    # re-runs the forward outside DDP's hooks, so gradients are "marked ready twice" and
    # DDP errors out. PEFT's own multi-GPU SFT guide calls this out. It costs some peak
    # memory (ample headroom at 96 GB/card) and is harmless single-GPU.
    gc_kwargs = {"use_reentrant": False}
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_ckpt,
            gradient_checkpointing_kwargs=gc_kwargs)
    elif grad_ckpt:
        # Non-quantized PEFT run: prepare_model_for_kbit_training isn't called, so enable
        # gradient checkpointing directly. enable_input_require_grads() is required for GC
        # to work with a frozen base + LoRA adapters (kbit prep does this internally).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()

    # PEFT task_type follows the loader that succeeded: SEQ_2_SEQ_LM for encoder-decoder
    # models, CAUSAL_LM otherwise (decoder-only LMs and text-generation VLMs).
    task_type = "SEQ_2_SEQ_LM" if used_cls_name == "AutoModelForSeq2SeqLM" else "CAUSAL_LM"
    target_modules = _resolve_target_modules(model, lora["target_modules"])
    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
        use_rslora=lora.get("use_rslora", False),
        use_dora=lora.get("use_dora", False),
    )
    model = get_peft_model(model, peft_cfg)
    # Second pass, AFTER adapter injection. target_modules matches leaf names, and VLM vision
    # towers reuse the language model's names (q_proj, gate_proj, ...), so LoRA lands inside
    # the frozen vision encoder too. Those adapters are trainable but never touched by a
    # text-only forward — under DDP that is an unused parameter and training errors out.
    # Freezing them also drops dead parameters from the adapter on single-GPU runs.
    _maybe_freeze_vision_encoder(model, cfg, stage="adapter")
    model.print_trainable_parameters()
    return LoadedModel(model, tokenizer, "peft")


def _maybe_freeze_vision_encoder(model: Any, cfg: dict, stage: str = "base") -> None:
    """Freeze vision encoder if present, controlled by cfg["freeze_vision_encoder"].

    "auto" (default) — freeze only if vision params are found.
    True             — always freeze (use if auto misses your VLM's naming).
    False            — skip entirely (pure LMs, avoids log noise).

    Called twice on the PEFT path: once on the raw model (freezes the base vision weights)
    and once after get_peft_model (freezes LoRA adapters injected into the vision tower).
    `stage` only labels the log line.
    """
    mode = cfg.get("freeze_vision_encoder", "auto")
    if mode is False or mode == "false":
        return
    frozen = 0
    found = False
    for name, param in model.named_parameters():
        if _is_vision_param(name):
            found = True
            if param.requires_grad:
                param.requires_grad_(False)
                frozen += 1
    if mode == "auto" and not found:
        return  # pure LM — silent
    if frozen:
        rank0_print(f"[_common] frozen {frozen} vision {stage} parameters")
    elif (mode is True or mode == "true") and stage == "base":
        rank0_print("[_common] WARNING: freeze_vision_encoder=true but no vision params found.")


_VISION_PARTS = frozenset({"visual", "vision_tower", "vision_encoder",
                           "img_projection", "video_projection"})


def _is_vision_param(name: str) -> bool:
    """True if any dotted path component names a vision submodule.

    Component matching (not startswith) so this keeps working after PEFT rewraps the model
    and parameter names gain a `base_model.model.` prefix.
    """
    return any(part in _VISION_PARTS for part in name.split("."))


def load_for_merge(cfg: dict) -> LoadedModel:
    """Load base model in bf16 + apply the trained adapter from output_dir for merging.

    Must be called instead of load_model_and_tokenizer when --merge is used.
    The base model is loaded in bf16 (not 4-bit) because merge_and_unload requires
    full-precision base weights to produce a clean merged checkpoint.
    """
    import importlib
    from transformers import AutoTokenizer
    from peft import PeftModel

    if dist_env()[2] > 1:
        raise SystemExit(
            "[_common] --merge must run as a single process — every rank would load the "
            "bf16 base and write the same merged/ directory concurrently. Re-run without "
            "the launcher, e.g. `python scripts/train/cpt.py --config <cfg> --merge` "
            "(or `make cpt-merge`).")

    _disable_broken_causal_conv1d()
    adapter_dir = cfg["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)

    # Merge is single-process by definition, so "auto" (shard the bf16 base across all
    # visible cards) is the right placement here.
    load_kwargs = dict(torch_dtype=_torch_dtype(cfg), device_map="auto",
                       trust_remote_code=True)
    model = None
    last_exc = None
    for cls_name in _MODEL_LOADERS:   # same arch-flexible order as the training loader
        try:
            cls = getattr(importlib.import_module("transformers"), cls_name)
            model = cls.from_pretrained(cfg["base_model"], **load_kwargs)
            rank0_print(f"[_common] merge base loaded with {cls_name}")
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).splitlines()[0] if str(exc) else ""
            rank0_print(f"[_common] {cls_name} failed ({exc.__class__.__name__}: {msg}); "
                        "trying next...")
    if model is None:
        raise RuntimeError(
            f"Could not load {cfg['base_model']} for merge with any Auto class. "
            f"Last error: {last_exc}"
        ) from last_exc
    _maybe_freeze_vision_encoder(model, cfg)
    model = PeftModel.from_pretrained(model, adapter_dir)
    rank0_print(f"[_common] loaded trained adapter from {adapter_dir}")
    return LoadedModel(model, tokenizer, "peft")


def merge_and_save(loaded: LoadedModel, out_dir: str) -> str:
    """Merge LoRA into full weights — produces input for the next stage or GGUF export."""
    merged_dir = str(Path(out_dir) / "merged")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    if loaded.backend == "unsloth":
        loaded.model.save_pretrained_merged(merged_dir, loaded.tokenizer,
                                            save_method="merged_16bit")
    else:
        merged = loaded.model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        loaded.tokenizer.save_pretrained(merged_dir)
    rank0_print(f"[_common] merged model -> {merged_dir}")
    return merged_dir
