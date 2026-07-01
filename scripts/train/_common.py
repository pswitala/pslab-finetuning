"""Shared helpers for the QLoRA training scripts (CPT / SFT / DPO).

Supports any HuggingFace causal-LM or VLM via PEFT. Tries Unsloth first;
falls back to PEFT + bitsandbytes automatically.

Config keys consumed here:
  base_model            — HF model ID or local path
  lora.target_modules   — projection name list, or "auto" for auto-detection
  freeze_vision_encoder — "auto" | true | false  (default: "auto")
  load_in_4bit          — NF4 quantization
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


_ATTN_PATTERNS = [
    ["q_proj", "k_proj", "v_proj", "o_proj"],   # Llama, Mistral, Qwen attn, Gemma
    ["c_attn", "c_proj"],                         # GPT-2 / GPT-NeoX (fused QKV)
    ["to_q", "to_k", "to_v", "to_out"],           # Diffusers / DiT
    ["query", "key", "value", "dense"],            # BERT / RoBERTa
    ["query_key_value", "dense"],                  # Falcon / older MPT
]
_MLP_PATTERNS = [
    ["gate_proj", "up_proj", "down_proj"],         # Llama / Mistral / Qwen MLP
    ["fc1", "fc2"],                                # GPT-2 / BERT MLP
    ["w1", "w2", "w3"],                            # some custom architectures
]
_SSM_CANDIDATES = [
    "in_proj_qkv", "out_proj", "in_proj_z", "in_proj_a", "in_proj_b",
    "in_proj", "x_proj", "dt_proj",               # Mamba / Qwen SSM
]


def _resolve_target_modules(model: Any, cfg_modules: Any) -> list[str] | Any:
    """Return resolved LoRA target_modules list.

    "auto" → inspect model.named_modules() for common projection patterns.
    list   → returned unchanged (user override).
    Falls back to ["q_proj", "v_proj"] if nothing detected.
    """
    if cfg_modules != "auto":
        return cfg_modules
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    detected: list[str] = []
    for pat in _ATTN_PATTERNS:
        hits = [p for p in pat if p in names]
        if hits:
            detected.extend(hits)
            break
    for pat in _MLP_PATTERNS:
        hits = [p for p in pat if p in names]
        if hits:
            detected.extend(hits)
            break
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
    print("[_common] WARNING: auto-detection found nothing; falling back to ['q_proj','v_proj']. "
          "Set lora.target_modules explicitly.")
    return ["q_proj", "v_proj"]


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    backend: str  # "unsloth" | "peft"


def load_model_and_tokenizer(cfg: dict) -> LoadedModel:
    """Load model + attach QLoRA adapter.

    Tries Unsloth first (fast kernels); falls back to PEFT + bitsandbytes
    if Unsloth does not support this architecture.
    """
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
        print(f"[_common] Unsloth unavailable for this model ({exc}); using PEFT fallback.")
        return _load_peft_fallback(cfg)


_MODEL_LOADERS = [
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForVision2Seq",
]


def _load_peft_fallback(cfg: dict) -> LoadedModel:
    """PEFT path — used when Unsloth is unavailable or does not support this architecture."""
    import importlib
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    model = None
    last_exc = None
    for cls_name in _MODEL_LOADERS:
        try:
            cls = getattr(importlib.import_module("transformers"), cls_name)
            model = cls.from_pretrained(cfg["base_model"], **load_kwargs)
            print(f"[_common] loaded with {cls_name}")
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[_common] {cls_name} failed ({exc.__class__.__name__}); trying next...")
    if model is None:
        raise RuntimeError(
            f"Could not load {cfg['base_model']} with any Auto class. "
            f"Last error: {last_exc}"
        ) from last_exc

    _maybe_freeze_vision_encoder(model, cfg)

    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))

    # PEFT task_type: CAUSAL_LM works for text generation models including VLMs.
    target_modules = _resolve_target_modules(model, lora["target_modules"])
    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_dir = cfg["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
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
