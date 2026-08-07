from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def compute_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    raise ValueError(f"Unsupported compute dtype: {name}")


def make_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig:
    quantization = config["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
        bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype(str(quantization["bnb_4bit_compute_dtype"])),
    )


def load_tokenizer(
    config: dict[str, Any],
    revision: str,
    cache_dir: str | Path,
    *,
    padding_side: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=revision,
        cache_dir=str(cache_dir),
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_generation_model(
    config: dict[str, Any],
    revision: str,
    cache_dir: str | Path,
    adapter_path: str | Path | None = None,
):
    dtype = compute_dtype(str(config["quantization"]["bnb_4bit_compute_dtype"]))
    use_4bit = bool(
        config.get("evaluation", {}).get(
            "load_in_4bit", config["quantization"]["load_in_4bit"]
        )
    )
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": str(cache_dir),
        "trust_remote_code": bool(config["model"]["trust_remote_code"]),
        "dtype": dtype,
        "device_map": {"": torch.cuda.current_device()},
        "attn_implementation": "sdpa",
    }
    if use_4bit:
        model_kwargs["quantization_config"] = make_quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        **model_kwargs,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model
