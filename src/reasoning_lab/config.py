from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


TOP_LEVEL_KEYS = {
    "project",
    "paths",
    "data",
    "model",
    "quantization",
    "lora",
    "training",
    "evaluation",
    "external_evaluation",
    "continuation",
    "stage3",
    "stage4",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "paths",
    "data",
    "model",
    "quantization",
    "lora",
    "training",
    "evaluation",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(_canonical_json(serializable).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")

    unknown = set(config) - TOP_LEVEL_KEYS
    missing = REQUIRED_TOP_LEVEL_KEYS - set(config)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing top-level configuration keys: {sorted(missing)}")

    _validate(config)
    project_root = config_path.parent.parent
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root)
    config["_config_hash"] = config_hash(config)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    raw = Path(config["paths"][key])
    if raw.is_absolute():
        return raw
    return Path(config["_project_root"]) / raw


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _validate(config: dict[str, Any]) -> None:
    for name in ("train_size", "validation_size", "test_size"):
        if int(config["data"].get(name, 0)) <= 0:
            raise ValueError(f"data.{name} must be positive")

    if int(config["training"].get("max_steps", 0)) <= 0:
        raise ValueError("training.max_steps must be positive for Stage 1")
    if int(config["training"].get("max_length", 0)) < 32:
        raise ValueError("training.max_length must be at least 32")
    if int(config["evaluation"].get("max_new_tokens", 0)) <= 0:
        raise ValueError("evaluation.max_new_tokens must be positive")
    if config["quantization"].get("bnb_4bit_quant_type") not in {"nf4", "fp4"}:
        raise ValueError("quantization.bnb_4bit_quant_type must be nf4 or fp4")
    if config["lora"].get("target_modules") != "all-linear":
        raise ValueError("Stage 1 requires QLoRA target_modules=all-linear")

    if "external_evaluation" in config:
        external = config["external_evaluation"]
        for name in ("fewshot_count", "parser_validation_size", "test_size"):
            if int(external.get(name, 0)) <= 0:
                raise ValueError(f"external_evaluation.{name} must be positive")
        if len(str(external.get("revision", ""))) != 40:
            raise ValueError("external_evaluation.revision must be a full 40-character commit SHA")
        for name in ("batch_size", "max_new_tokens", "bootstrap_samples"):
            if int(external.get(name, 0)) <= 0:
                raise ValueError(f"external_evaluation.{name} must be positive")
    if "continuation" in config:
        continuation = config["continuation"]
        if int(config["data"].get("continuation_size", 0)) <= 0:
            raise ValueError("data.continuation_size must be positive when continuation is configured")
        if int(continuation.get("max_steps", 0)) <= 0:
            raise ValueError("continuation.max_steps must be positive")
    if "stage3" in config:
        from reasoning_lab.stage3_selection import validate_stage3_spec

        stage3 = config["stage3"]
        for name in (
            "synthetic_dev_size",
            "synthetic_confirm_size",
            "external_dev_size",
            "external_confirm_size",
        ):
            if int(stage3.get(name, 0)) <= 0:
                raise ValueError(f"stage3.{name} must be positive")
        validate_stage3_spec(stage3)
    if "stage4" in config:
        from reasoning_lab.stage4_selection import validate_stage4_spec

        validate_stage4_spec(config["stage4"])
