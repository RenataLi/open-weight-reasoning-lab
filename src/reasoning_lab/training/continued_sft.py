from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json
from reasoning_lab.models.loading import (
    compute_dtype,
    load_tokenizer,
    make_quantization_config,
)


EXACT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _disable_dropout_modules(model: torch.nn.Module) -> list[float]:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return sorted(
        {
            float(module.p)
            for module in model.modules()
            if isinstance(module, torch.nn.Dropout)
        }
    )


def sha256_directory(path: str | Path) -> str:
    """Hash a directory without depending on host-specific absolute paths."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {root}")

    files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"Adapter directory is empty: {root}")

    digest = hashlib.sha256()
    for file_path in files:
        relative_path = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_parent_lineage(
    parent_adapter: str | Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, str]:
    """Validate that the parent adapter records the exact base snapshot."""

    if not EXACT_REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            "continued-SFT requires an immutable 40-character model commit revision"
        )

    adapter_root = Path(parent_adapter)
    config_path = adapter_root / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing parent adapter config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    recorded_model = str(adapter_config.get("base_model_name_or_path") or "")
    recorded_revision = str(adapter_config.get("revision") or "")
    if recorded_model != model_id:
        raise ValueError(
            "Parent adapter base model mismatch: "
            f"expected {model_id!r}, recorded {recorded_model!r}"
        )
    if recorded_revision.lower() != revision.lower():
        raise ValueError(
            "Parent adapter base revision mismatch: "
            f"expected {revision!r}, recorded {recorded_revision or None!r}"
        )

    return {
        "base_model_id": recorded_model,
        "base_model_revision": recorded_revision.lower(),
        "adapter_sha256": sha256_directory(adapter_root),
    }


def _require_fresh_output(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_root = Path(output_dir)
    checkpoint_root = output_root / "checkpoints"
    final_adapter = output_root / "final_adapter"
    summary_path = output_root / "training_summary.json"

    occupied = [path for path in (checkpoint_root, final_adapter, summary_path) if path.exists()]
    if occupied:
        joined = ", ".join(str(path) for path in occupied)
        raise FileExistsError(
            "continued-SFT is a fresh branch and never resumes or overwrites state; "
            f"choose a new output directory (found: {joined})"
        )
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    return output_root, checkpoint_root, final_adapter


def _continuation_dataset(path: str | Path) -> Dataset:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Continuation dataset is empty")

    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Continuation row {index} has no non-empty prompt")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"Continuation row {index} has no non-empty completion")
        normalized.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(normalized)


def _continuation_settings(config: dict[str, Any]) -> dict[str, Any]:
    if "continuation" not in config:
        raise KeyError("Missing continuation configuration")
    base = config["training"]
    branch = config["continuation"]
    keys = {
        "seed",
        "data_seed",
        "max_length",
        "max_steps",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_ratio",
        "weight_decay",
        "gradient_checkpointing",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "save_strategy",
        "loss_type",
    }
    settings = {key: branch.get(key, base.get(key)) for key in keys}
    settings["loss_type"] = branch.get("loss_type", base.get("loss_type", "nll"))
    settings["save_strategy"] = branch.get(
        "save_strategy", base.get("save_strategy", "steps")
    )
    settings["disable_dropout"] = bool(
        branch.get("disable_dropout", base.get("disable_dropout", False))
    )
    missing = sorted(key for key, value in settings.items() if value is None)
    if missing:
        raise KeyError(f"Missing continued-SFT settings: {missing}")
    if int(settings["max_steps"]) <= 0:
        raise ValueError("continuation.max_steps must be positive")
    return settings


def _train_from_fresh_state(trainer: SFTTrainer):
    """Start a new optimizer/scheduler/global-step trajectory, never a resume."""

    if int(trainer.state.global_step) != 0:
        raise RuntimeError("Fresh continued-SFT trainer must start at global_step=0")
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise RuntimeError("Fresh continued-SFT trainer must not receive optimizer state")
    return trainer.train(resume_from_checkpoint=None)


def train_continued_sft(
    config: dict[str, Any],
    *,
    revision: str,
    cache_dir: str | Path,
    continuation_path: str | Path,
    parent_adapter: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train a continued-SFT branch from a parent PEFT adapter with fresh state."""

    parent_root = Path(parent_adapter).resolve()
    lineage = validate_parent_lineage(
        parent_root,
        model_id=str(config["model"]["id"]),
        revision=revision,
    )
    settings = _continuation_settings(config)
    dataset = _continuation_dataset(continuation_path)
    requested_output = Path(output_dir).resolve()
    if (
        requested_output == parent_root
        or requested_output in parent_root.parents
        or parent_root in requested_output.parents
    ):
        raise ValueError("Parent adapter and continued-SFT output must not be nested")
    output_root, checkpoint_root, final_adapter = _require_fresh_output(requested_output)
    tokenizer = load_tokenizer(config, revision, cache_dir, padding_side="right")
    dtype = compute_dtype(str(config["quantization"]["bnb_4bit_compute_dtype"]))

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        revision=revision,
        cache_dir=str(cache_dir),
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        quantization_config=make_quantization_config(config),
        dtype=dtype,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=bool(settings["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(base_model, str(parent_root), is_trainable=True)
    model.config.use_cache = False
    if bool(settings["disable_dropout"]):
        effective_dropout_probabilities = _disable_dropout_modules(model)
    else:
        effective_dropout_probabilities = sorted(
            {
                float(module.p)
                for module in model.modules()
                if isinstance(module, torch.nn.Dropout)
            }
        )

    warmup_steps = round(int(settings["max_steps"]) * float(settings["warmup_ratio"]))
    training_args = SFTConfig(
        output_dir=str(checkpoint_root),
        max_length=int(settings["max_length"]),
        packing=False,
        completion_only_loss=True,
        loss_type=str(settings.get("loss_type", "nll")),
        learning_rate=float(settings["learning_rate"]),
        max_steps=int(settings["max_steps"]),
        num_train_epochs=float(settings["num_train_epochs"]),
        per_device_train_batch_size=int(settings["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        lr_scheduler_type=str(settings["lr_scheduler_type"]),
        warmup_steps=warmup_steps,
        weight_decay=float(settings["weight_decay"]),
        optim="adamw_torch_fused",
        gradient_checkpointing=bool(settings["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        tf32=True,
        logging_steps=int(settings["logging_steps"]),
        logging_first_step=True,
        save_strategy=str(settings.get("save_strategy", "steps")),
        save_steps=int(settings["save_steps"]),
        save_total_limit=int(settings["save_total_limit"]),
        save_only_model=False,
        report_to="none",
        push_to_hub=False,
        seed=int(settings["seed"]),
        data_seed=int(settings["data_seed"]),
        dataset_num_proc=None,
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(None, None),
    )

    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    if trainable_parameters <= 0 or trainable_parameters >= total_parameters:
        raise RuntimeError("Expected a non-zero strict subset of parameters to be trainable")
    trainable_names = [
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    ]
    if any(".default." not in name for name in trainable_names):
        raise RuntimeError("continued-SFT has trainable parameters outside the default adapter")
    trainable_names_sha256 = hashlib.sha256(
        "\n".join(sorted(trainable_names)).encode("utf-8")
    ).hexdigest()
    configured_lora_dropout = float(
        trainer.model.peft_config["default"].lora_dropout
    )
    if bool(settings["disable_dropout"]) and any(
        value != 0.0 for value in effective_dropout_probabilities
    ):
        raise RuntimeError("continued-SFT dropout disabling was incomplete")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = _train_from_fresh_state(trainer)
    if not torch.isfinite(torch.tensor(float(result.training_loss))):
        raise RuntimeError("Training loss is NaN or infinite")

    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    summary = {
        "training_kind": "continued_sft_branch",
        "global_step": int(trainer.state.global_step),
        "initial_global_step": 0,
        "training_loss": float(result.training_loss),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "trainable_parameter_names_sha256": trainable_names_sha256,
        "fresh_optimizer": True,
        "fresh_scheduler": True,
        "resumed_from_checkpoint": None,
        "base_model_id": str(config["model"]["id"]),
        "base_model_revision": revision.lower(),
        "parent_adapter": str(parent_root),
        "parent_adapter_sha256": lineage["adapter_sha256"],
        "parent_base_model_id": lineage["base_model_id"],
        "parent_base_model_revision": lineage["base_model_revision"],
        "continuation_dataset": str(Path(continuation_path).resolve()),
        "continuation_examples": len(dataset),
        "continuation_dataset_sha256": sha256_file(continuation_path),
        "resolved_training_settings": {
            "data_seed": int(settings["data_seed"]),
            "learning_rate": float(settings["learning_rate"]),
            "loss_type": str(settings.get("loss_type", "nll")),
            "max_length": int(settings["max_length"]),
            "max_steps": int(settings["max_steps"]),
            "seed": int(settings["seed"]),
            "disable_dropout": bool(settings["disable_dropout"]),
            "configured_lora_dropout": configured_lora_dropout,
            "effective_dropout_probabilities": effective_dropout_probabilities,
        },
        "peak_gpu_memory_bytes": peak_memory,
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "final_adapter": str(final_adapter.resolve()),
        "final_adapter_sha256": sha256_directory(final_adapter),
    }
    write_json(output_root / "training_summary.json", summary)

    del trainer, model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
