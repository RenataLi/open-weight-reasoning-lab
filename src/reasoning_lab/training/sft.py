from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json
from reasoning_lab.models.loading import (
    compute_dtype,
    load_tokenizer,
    make_quantization_config,
)
from reasoning_lab.training.continued_sft import sha256_directory


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    if not root.exists():
        return None
    for path in root.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def train_sft(
    config: dict[str, Any],
    *,
    revision: str,
    cache_dir: str | Path,
    train_path: str | Path,
    output_dir: str | Path,
    resume: bool = True,
    require_fresh: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    if require_fresh and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Fresh SFT output directory is not empty: {output_root}")
    checkpoint_root = output_root / "checkpoints"
    final_adapter = output_root / "final_adapter"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        config,
        revision,
        cache_dir,
        padding_side="right",
    )
    rows = read_jsonl(train_path)
    if not rows:
        raise ValueError("SFT training data is empty")
    completion_token_count = sum(
        len(tokenizer(row["completion"], add_special_tokens=False)["input_ids"])
        for row in rows
    )
    prompt_token_count = sum(
        len(tokenizer(row["prompt"], add_special_tokens=True)["input_ids"])
        for row in rows
    )
    dataset = Dataset.from_list(
        [{"prompt": row["prompt"], "completion": row["completion"]} for row in rows]
    )

    lora = config["lora"]
    peft_config = LoraConfig(
        task_type=str(lora["task_type"]),
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        target_modules=str(lora["target_modules"]),
        revision=revision,
    )

    training = config["training"]
    dtype = compute_dtype(str(config["quantization"]["bnb_4bit_compute_dtype"]))
    warmup_steps = round(int(training["max_steps"]) * float(training["warmup_ratio"]))
    training_args = SFTConfig(
        output_dir=str(checkpoint_root),
        model_init_kwargs={
            "revision": revision,
            "cache_dir": str(cache_dir),
            "dtype": dtype,
            "trust_remote_code": bool(config["model"]["trust_remote_code"]),
            "attn_implementation": "sdpa",
        },
        max_length=int(training["max_length"]),
        packing=False,
        completion_only_loss=bool(training.get("completion_only_loss", True)),
        loss_type=str(training.get("loss_type", "nll")),
        learning_rate=float(training["learning_rate"]),
        max_steps=int(training["max_steps"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        warmup_steps=warmup_steps,
        weight_decay=float(training["weight_decay"]),
        optim="adamw_torch_fused",
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        bf16=True,
        fp16=False,
        tf32=True,
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        save_strategy=str(training.get("save_strategy", "steps")),
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        save_only_model=False,
        report_to="none",
        push_to_hub=False,
        seed=int(training["seed"]),
        data_seed=int(training["data_seed"]),
        dataset_num_proc=None,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=config["model"]["id"],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        quantization_config=make_quantization_config(config),
        peft_config=peft_config,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    if trainable_parameters <= 0 or trainable_parameters >= total_parameters:
        raise RuntimeError("Expected a non-zero strict subset of parameters to be trainable")

    resume_checkpoint = latest_checkpoint(checkpoint_root) if resume else None
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    if not torch.isfinite(torch.tensor(float(result.training_loss))):
        raise RuntimeError("Training loss is NaN or infinite")

    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    resolved_settings = {
        "completion_only_loss": bool(training.get("completion_only_loss", True)),
        "data_seed": int(training["data_seed"]),
        "learning_rate": float(training["learning_rate"]),
        "loss_type": str(training.get("loss_type", "nll")),
        "max_length": int(training["max_length"]),
        "max_steps": int(training["max_steps"]),
        "seed": int(training["seed"]),
    }
    summary = {
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "final_adapter": str(final_adapter),
        "final_adapter_sha256": sha256_directory(final_adapter),
        "resolved_training_settings": resolved_settings,
        "training_data": {
            "path": str(Path(train_path).resolve()),
            "sha256": sha256_file(train_path),
            "count": len(rows),
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
        },
    }
    write_json(output_root / "training_summary.json", summary)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return summary
