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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def wandb_report_to() -> list[str]:
    """['wandb'] if WANDB_API_KEY is set, else [] — for SFTConfig/DPOConfig report_to.

    Shared by cpt/sft/dpo so experiment tracking is enabled consistently across stages.
    """
    import os
    return ["wandb"] if os.environ.get("WANDB_API_KEY") else []


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
        print(f"[_common] auto-detected target_modules: {unique}")
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
        print("[_common] Unsloth disabled (use_unsloth=false / PSLAB_NO_UNSLOTH); "
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
        print("[_common] " + "=" * 68)
        print(f"[_common] Unsloth path failed ({exc.__class__.__name__}: {exc}); "
              "falling back to PEFT + bitsandbytes.")
        traceback.print_exc()
        print("[_common] " + "=" * 68)
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
        print(f"[_common] broken/absent causal_conv1d ({exc.__class__.__name__}); "
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
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
                print(f"[_common] loaded with {cls_name} (attn_implementation={attn_impl})")
                return m, cls_name, None
            except Exception as exc:  # noqa: BLE001
                exc_seen = exc
                msg = str(exc).splitlines()[0] if str(exc) else ""
                print(f"[_common] {cls_name} failed (attn={attn_impl}) "
                      f"({exc.__class__.__name__}: {msg}); trying next...")
        return None, None, exc_seen

    model, used_cls_name, last_exc = _try_loaders(attn_pref)
    if model is None and attn_pref != "eager" and _is_attn_impl_error(last_exc):
        print(f"[_common] attn_implementation={attn_pref} unsupported here; retrying with eager")
        model, used_cls_name, last_exc = _try_loaders("eager")
    if model is None:
        raise RuntimeError(
            f"Could not load {cfg['base_model']} with any Auto class. "
            f"Last error: {last_exc}"
        ) from last_exc

    _maybe_freeze_vision_encoder(model, cfg)

    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))

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
    model.print_trainable_parameters()
    return LoadedModel(model, tokenizer, "peft")


def _maybe_freeze_vision_encoder(model: Any, cfg: dict) -> None:
    """Freeze vision encoder if present, controlled by cfg["freeze_vision_encoder"].

    "auto" (default) — freeze only if vision-prefixed params are found.
    True             — always freeze (use if auto misses your VLM's naming).
    False            — skip entirely (pure LMs, avoids log noise).
    """
    mode = cfg.get("freeze_vision_encoder", "auto")
    if mode is False or mode == "false":
        return
    frozen = 0
    found = False
    for name, param in model.named_parameters():
        if _is_vision_param(name):
            found = True
            param.requires_grad_(False)
            frozen += 1
    if mode == "auto" and not found:
        return  # pure LM — silent
    if frozen:
        print(f"[_common] frozen {frozen} vision encoder parameters")
    elif mode is True or mode == "true":
        print("[_common] WARNING: freeze_vision_encoder=true but no vision params found.")


def _is_vision_param(name: str) -> bool:
    vision_prefixes = ("visual.", "vision_tower.", "vision_encoder.",
                       "img_projection.", "video_projection.")
    return any(name.startswith(p) for p in vision_prefixes)


def load_for_merge(cfg: dict) -> LoadedModel:
    """Load base model in bf16 + apply the trained adapter from output_dir for merging.

    Must be called instead of load_model_and_tokenizer when --merge is used.
    The base model is loaded in bf16 (not 4-bit) because merge_and_unload requires
    full-precision base weights to produce a clean merged checkpoint.
    """
    import importlib
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    _disable_broken_causal_conv1d()
    adapter_dir = cfg["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto",
                       trust_remote_code=True)
    model = None
    last_exc = None
    for cls_name in _MODEL_LOADERS:   # same arch-flexible order as the training loader
        try:
            cls = getattr(importlib.import_module("transformers"), cls_name)
            model = cls.from_pretrained(cfg["base_model"], **load_kwargs)
            print(f"[_common] merge base loaded with {cls_name}")
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).splitlines()[0] if str(exc) else ""
            print(f"[_common] {cls_name} failed ({exc.__class__.__name__}: {msg}); trying next...")
    if model is None:
        raise RuntimeError(
            f"Could not load {cfg['base_model']} for merge with any Auto class. "
            f"Last error: {last_exc}"
        ) from last_exc
    _maybe_freeze_vision_encoder(model, cfg)
    model = PeftModel.from_pretrained(model, adapter_dir)
    print(f"[_common] loaded trained adapter from {adapter_dir}")
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
    print(f"[_common] merged model -> {merged_dir}")
    return merged_dir
