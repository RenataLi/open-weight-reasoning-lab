from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json
from reasoning_lab.models.loading import compute_dtype, load_tokenizer, make_quantization_config
from reasoning_lab.preference_tokenization import trl_preference_token_ids
from reasoning_lab.training.continued_sft import (
    sha256_directory,
    validate_parent_lineage,
)


SUPPORTED_DPO_LOSSES = {"sigmoid", "sigmoid_norm"}


def _preference_dataset(path: str | Path) -> Dataset:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Preference dataset is empty")
    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or row.get("pair_id") or "")
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not row_id:
            raise ValueError(f"Preference row {index} has no id")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate preference id: {row_id}")
        if not all(isinstance(value, str) and value.strip() for value in (prompt, chosen, rejected)):
            raise ValueError(f"Preference row {row_id} has an empty text field")
        if chosen == rejected:
            raise ValueError(f"Preference row {row_id} has identical completions")
        seen_ids.add(row_id)
        normalized.append(
            {"id": row_id, "prompt": prompt, "chosen": chosen, "rejected": rejected}
        )
    return Dataset.from_list(normalized)


def _require_fresh_output(output_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_dir).resolve()
    checkpoints = root / "checkpoints"
    final_adapter = root / "final_adapter"
    summary = root / "training_summary.json"
    occupied = [path for path in (checkpoints, final_adapter, summary) if path.exists()]
    if occupied:
        raise FileExistsError(
            "DPO always starts a fresh branch; choose a new output directory "
            f"(found: {', '.join(str(path) for path in occupied)})"
        )
    checkpoints.mkdir(parents=True, exist_ok=False)
    return root, checkpoints, final_adapter


def _adapter_tensor_hash(model: PeftModel, adapter_name: str) -> str:
    marker = f".{adapter_name}."
    tensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if marker in name
    ]
    if not tensors:
        raise ValueError(f"Adapter {adapter_name!r} has no parameters")
    digest = hashlib.sha256()
    for name, parameter in sorted(tensors):
        canonical_name = name.replace(marker, ".<adapter>.")
        value = parameter.detach().cpu().contiguous()
        digest.update(canonical_name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_reference_clone(model: PeftModel) -> dict[str, Any]:
    if "default" not in model.peft_config or "ref" not in model.peft_config:
        raise RuntimeError("TRL did not create the expected default/ref adapter pair")
    named = dict(model.named_parameters())
    compared = 0
    max_abs_diff = 0.0
    for name, parameter in named.items():
        if ".default." not in name:
            continue
        ref_name = name.replace(".default.", ".ref.")
        if ref_name not in named:
            raise RuntimeError(f"Missing reference tensor corresponding to {name}")
        reference = named[ref_name]
        compared += 1
        if reference.requires_grad:
            raise RuntimeError(f"Reference adapter tensor is trainable: {ref_name}")
        if parameter.shape != reference.shape:
            raise RuntimeError(f"Policy/reference shape mismatch for {name}")
        difference = float((parameter.detach() - reference.detach()).abs().max().item())
        max_abs_diff = max(max_abs_diff, difference)
    if compared == 0:
        raise RuntimeError("No default adapter tensors were compared")
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Policy/reference adapters differ at initialization: {max_abs_diff}")
    return {
        "compared_tensor_count": compared,
        "max_abs_diff": max_abs_diff,
        "reference_sha256": _adapter_tensor_hash(model, "ref"),
        "policy_initial_sha256": _adapter_tensor_hash(model, "default"),
    }


def _assert_no_truncation(
    dataset: Dataset,
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, int]:
    longest_prompt = 0
    longest_sequence = 0
    for row in dataset:
        prompt_ids: list[int] | None = None
        for field in ("chosen", "rejected"):
            try:
                current_prompt_ids, _, full_ids = trl_preference_token_ids(
                    tokenizer,
                    prompt=row["prompt"],
                    completion=row[field],
                )
            except ValueError as error:
                raise ValueError(
                    f"{error} in preference row {row['id']} ({field})"
                ) from error
            if prompt_ids is None:
                prompt_ids = current_prompt_ids
                longest_prompt = max(longest_prompt, len(prompt_ids))
            elif prompt_ids != current_prompt_ids:
                raise RuntimeError("Prompt tokenization changed within a preference pair")
            longest_sequence = max(longest_sequence, len(full_ids))
            if len(full_ids) > max_length:
                raise ValueError(
                    f"Preference row {row['id']} has {len(full_ids)} tokens, "
                    f"exceeding max_length={max_length}; refusing truncation"
                )
    return {
        "longest_prompt_tokens": longest_prompt,
        "longest_sequence_tokens": longest_sequence,
    }


def train_dpo_branch(
    config: dict[str, Any],
    *,
    candidate: dict[str, Any],
    revision: str,
    cache_dir: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    parent_adapter: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one fresh DPO branch from an immutable pretrained LoRA adapter."""

    loss_type = str(candidate["loss_type"])
    if loss_type not in SUPPORTED_DPO_LOSSES:
        raise ValueError(f"Unsupported Stage 4 DPO loss: {loss_type}")
    auxiliary_loss_type = candidate.get("auxiliary_loss_type")
    loss_types = [loss_type]
    if auxiliary_loss_type is not None:
        if str(auxiliary_loss_type) != "sft":
            raise ValueError("Stage 4 only supports 'sft' as an auxiliary DPO loss")
        loss_types.append("sft")
    loss_weights_raw = candidate.get("loss_weights")
    loss_weights = (
        None
        if loss_weights_raw is None
        else [float(value) for value in loss_weights_raw]
    )
    if loss_weights is not None and (
        len(loss_weights) != len(loss_types)
        or any(value <= 0.0 for value in loss_weights)
    ):
        raise ValueError("Stage 4 DPO loss_weights must be positive and match loss_types")
    parent_root = Path(parent_adapter).resolve()
    lineage = validate_parent_lineage(
        parent_root,
        model_id=str(config["model"]["id"]),
        revision=revision,
    )
    requested_output = Path(output_dir).resolve()
    if (
        requested_output == parent_root
        or requested_output in parent_root.parents
        or parent_root in requested_output.parents
    ):
        raise ValueError("Parent adapter and DPO output must not be nested")
    output_root, checkpoint_root, final_adapter = _require_fresh_output(requested_output)

    settings = config["stage4"]["dpo_training"]
    disable_dropout = bool(settings.get("disable_dropout", True))
    if not disable_dropout:
        raise ValueError("Stage 4 DPO protocol requires deterministic dropout disabling")
    max_length = int(settings["max_length"])
    train_dataset = _preference_dataset(train_path)
    validation_dataset = _preference_dataset(validation_path)
    tokenizer = load_tokenizer(config, revision, cache_dir, padding_side="right")
    train_lengths = _assert_no_truncation(
        train_dataset, tokenizer, max_length=max_length
    )
    validation_lengths = _assert_no_truncation(
        validation_dataset, tokenizer, max_length=max_length
    )
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

    seed = int(candidate.get("seed", settings["seed"]))
    data_seed = int(candidate.get("data_seed", settings.get("data_seed", seed)))
    max_steps = int(candidate.get("max_steps", settings["max_steps"]))
    learning_rate = float(candidate.get("learning_rate", settings["learning_rate"]))
    beta = float(candidate.get("beta", settings["beta"]))
    warmup_steps = int(candidate.get("warmup_steps", settings.get("warmup_steps", 1)))
    args = DPOConfig(
        output_dir=str(checkpoint_root),
        max_length=max_length,
        truncation_mode="keep_start",
        padding_free=False,
        beta=beta,
        loss_type=loss_types,
        loss_weights=loss_weights,
        precompute_ref_log_probs=bool(settings.get("precompute_ref_log_probs", True)),
        precompute_ref_batch_size=int(settings.get("precompute_ref_batch_size", 8)),
        sync_ref_model=False,
        # DPO reference log-probabilities must be deterministic. Stage 4 controls
        # explicitly disable the same inherited adapter dropout for a fair comparison.
        disable_dropout=disable_dropout,
        learning_rate=learning_rate,
        max_steps=max_steps,
        num_train_epochs=float(settings.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(settings["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        per_device_eval_batch_size=int(settings.get("per_device_eval_batch_size", 8)),
        lr_scheduler_type=str(settings.get("lr_scheduler_type", "linear")),
        warmup_steps=warmup_steps,
        weight_decay=float(settings.get("weight_decay", 0.0)),
        max_grad_norm=float(settings.get("max_grad_norm", 1.0)),
        optim="adamw_torch_fused",
        gradient_checkpointing=bool(settings["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        tf32=True,
        use_liger_kernel=False,
        logging_steps=int(settings.get("logging_steps", 2)),
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        push_to_hub=False,
        seed=seed,
        data_seed=data_seed,
        dataset_num_proc=None,
        dataloader_num_workers=0,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        optimizers=(None, None),
    )
    clone = _verify_reference_clone(trainer.model)
    reference_hash_before = clone["reference_sha256"]
    trainable_names = [
        name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(".default." not in name for name in trainable_names):
        raise RuntimeError("Unexpected trainable parameter set in DPO branch")
    trainable_names_sha256 = hashlib.sha256(
        "\n".join(sorted(trainable_names)).encode("utf-8")
    ).hexdigest()
    configured_lora_dropout = float(
        trainer.model.peft_config["default"].lora_dropout
    )
    effective_dropout_probabilities = sorted(
        {
            float(module.p)
            for module in trainer.model.modules()
            if isinstance(module, torch.nn.Dropout)
        }
    )
    if any(value != 0.0 for value in effective_dropout_probabilities):
        raise RuntimeError("DPO dropout disabling did not cover every dropout module")

    if int(trainer.state.global_step) != 0 or trainer.optimizer is not None:
        raise RuntimeError("Fresh DPO trainer must start without optimizer/global-step state")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=None)
    if not torch.isfinite(torch.tensor(float(result.training_loss))):
        raise RuntimeError("DPO training loss is NaN or infinite")
    eval_metrics = trainer.evaluate(metric_key_prefix="preference_validation")
    if not all(
        torch.isfinite(torch.tensor(float(value)))
        for value in eval_metrics.values()
        if isinstance(value, (int, float))
    ):
        raise RuntimeError("DPO validation produced a NaN or infinite metric")

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    reference_hash_after = _adapter_tensor_hash(unwrapped, "ref")
    if reference_hash_after != reference_hash_before:
        raise RuntimeError("Frozen DPO reference adapter changed during training")
    unwrapped.save_pretrained(str(final_adapter), selected_adapters=["default"])
    tokenizer.save_pretrained(str(final_adapter))
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    summary = {
        "training_kind": "dpo_from_pretrained_adapter",
        "candidate": {
            "id": str(candidate["id"]),
            "loss_type": loss_type,
            "loss_types": loss_types,
            "loss_weights": loss_weights,
            "beta": beta,
        },
        "global_step": int(trainer.state.global_step),
        "initial_global_step": 0,
        "training_loss": float(result.training_loss),
        "fresh_optimizer": True,
        "fresh_scheduler": True,
        "resumed_from_checkpoint": None,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "trainable_parameter_names_sha256": trainable_names_sha256,
        "base_model_id": str(config["model"]["id"]),
        "base_model_revision": revision.lower(),
        "parent_adapter": str(parent_root),
        "parent_adapter_sha256": lineage["adapter_sha256"],
        "parent_base_model_id": lineage["base_model_id"],
        "parent_base_model_revision": lineage["base_model_revision"],
        "reference_clone": clone,
        "reference_sha256_after": reference_hash_after,
        "reference_unchanged": True,
        "train_dataset": str(Path(train_path).resolve()),
        "train_dataset_sha256": sha256_file(train_path),
        "train_examples": len(train_dataset),
        "validation_dataset": str(Path(validation_path).resolve()),
        "validation_dataset_sha256": sha256_file(validation_path),
        "validation_examples": len(validation_dataset),
        "token_lengths": {
            "train": train_lengths,
            "validation": validation_lengths,
        },
        "resolved_training_settings": {
            "seed": seed,
            "data_seed": data_seed,
            "learning_rate": learning_rate,
            "beta": beta,
            "loss_type": loss_type,
            "loss_types": loss_types,
            "loss_weights": loss_weights,
            "max_length": max_length,
            "max_steps": max_steps,
            "effective_prompt_batch_size": int(settings["per_device_train_batch_size"])
            * int(settings["gradient_accumulation_steps"]),
            "precompute_ref_log_probs": bool(
                settings.get("precompute_ref_log_probs", True)
            ),
            "disable_dropout": disable_dropout,
            "configured_lora_dropout": configured_lora_dropout,
            "effective_dropout_probabilities": effective_dropout_probabilities,
        },
        "preference_validation_metrics": eval_metrics,
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
