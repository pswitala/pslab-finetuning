"""Shared helpers for the QLoRA training scripts (CPT / SFT / DPO).

Architecture facts for Qwen3.6-27B (verified 2026-06):
- Model type: qwen3_5  (Qwen3_5ForConditionalGeneration) — a VLM.
- Hybrid: 16 × (3× Gated DeltaNet + 1× Gated Attention) = 64 layers.
- 75% of layers are SSM/linear-attention (linear_attn.*).
- Vision encoder is present but frozen for text-only fine-tuning.
- No base checkpoint — only instruct. Thinking mode disabled via chat template.

Unsloth note: FastQwen3_5Model is NOT confirmed in Unsloth as of 2026-06 planning.
_load_peft_fallback is the expected primary path unless Unsloth adds support.
Check `unsloth --version` release notes before assuming the Unsloth path works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    backend: str  # "unsloth" | "peft"


def load_model_and_tokenizer(cfg: dict) -> LoadedModel:
    """Load Qwen3.6-27B + attach QLoRA adapter.

    Tries Unsloth first (fast kernels). Falls back to PEFT if Unsloth doesn't yet
    support the qwen3_5 architecture — which is expected to be the common case
    until Unsloth ships a Qwen3.6 release.
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
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora.get("dropout", 0.0),
            target_modules=lora["target_modules"],
            use_rslora=lora.get("use_rslora", False),
            use_dora=lora.get("use_dora", False),
            use_gradient_checkpointing="unsloth" if cfg.get("gradient_checkpointing", True) else False,
            random_state=cfg.get("seed", 3407),
        )
        _freeze_vision_encoder(model)
        return LoadedModel(model, tokenizer, "unsloth")
    except Exception as exc:  # noqa: BLE001
        print(f"[_common] Unsloth unavailable for this model ({exc}); using PEFT fallback.")
        return _load_peft_fallback(cfg)


def _load_peft_fallback(cfg: dict) -> LoadedModel:
    """PEFT path — primary path for Qwen3.6-27B until Unsloth adds qwen3_5 support.

    Qwen3_5ForConditionalGeneration is a VLM. We load it with AutoModelForCausalLM
    (works for text generation since the class supports it), then freeze the vision
    encoder so only text layers receive gradients.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

    # AutoModelForCausalLM works for text generation with Qwen VLMs.
    # If it fails for this model type, fall back to the explicit class.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            quantization_config=quant,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            cfg["base_model"],
            quantization_config=quant,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    _freeze_vision_encoder(model)

    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))

    # PEFT task_type: CAUSAL_LM works for text generation models including VLMs.
    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=lora.get("use_rslora", False),
        use_dora=lora.get("use_dora", False),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return LoadedModel(model, tokenizer, "peft")


def _freeze_vision_encoder(model: Any) -> None:
    """Freeze all vision encoder parameters — text-only fine-tuning."""
    frozen = 0
    for name, param in model.named_parameters():
        if _is_vision_param(name):
            param.requires_grad_(False)
            frozen += 1
    if frozen:
        print(f"[_common] frozen {frozen} vision encoder parameters")


def _is_vision_param(name: str) -> bool:
    vision_prefixes = ("visual.", "vision_tower.", "vision_encoder.",
                       "img_projection.", "video_projection.")
    return any(name.startswith(p) for p in vision_prefixes)


def load_for_merge(cfg: dict) -> LoadedModel:
    """Load base model in bf16 + apply the TRAINED adapter from output_dir for merging.

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
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _freeze_vision_encoder(model)
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
