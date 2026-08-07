from __future__ import annotations

import gc
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from reasoning_lab.models.loading import load_generation_model, load_tokenizer
from reasoning_lab.preference_tokenization import trl_preference_token_ids
from reasoning_lab.training.continued_sft import sha256_directory


def _masked_sequence_logps(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return completion log-prob sums and token counts for a causal LM batch."""

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = input_ids[:, 1:]
    shifted_mask = completion_mask[:, 1:].to(dtype=torch.bool)
    token_logps = torch.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_logps = token_logps.masked_fill(~shifted_mask, 0.0)
    counts = shifted_mask.sum(dim=1).clamp(min=1)
    return token_logps.sum(dim=1), counts


def _encode_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    completion_lengths: list[int] = []
    for field in ("chosen", "rejected"):
        for row in rows:
            try:
                prompt_ids, completion_ids, full_ids = trl_preference_token_ids(
                    tokenizer,
                    prompt=str(row["prompt"]),
                    completion=str(row[field]),
                )
            except ValueError as error:
                raise ValueError(
                    f"{error} in preference row {row['id']} ({field})"
                ) from error
            if len(full_ids) > max_length:
                raise ValueError(
                    f"Preference row {row['id']} exceeds max_length={max_length}"
                )
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            sequences.append(full_ids)
            masks.append(mask)
            completion_lengths.append(len(completion_ids))
    width = max(len(ids) for ids in sequences)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = []
    attention_mask = []
    completion_mask = []
    for ids, mask in zip(sequences, masks, strict=True):
        padding = width - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        completion_mask.append(mask + [0] * padding)
    encoded = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.long),
    }
    return encoded, completion_lengths


def _summarize(rows: list[dict[str, Any]], margin_key: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty preference results")

    def one(items: list[dict[str, Any]]) -> dict[str, Any]:
        margins = [float(row[margin_key]) for row in items]
        return {
            "count": len(items),
            "reward_accuracy": sum(value > 0.0 for value in margins) / len(margins),
            "reward_tie_rate": sum(value == 0.0 for value in margins) / len(margins),
            "reward_margin_mean": statistics.fmean(margins),
            "reward_margin_median": statistics.median(margins),
        }

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family", "unknown"))].append(row)
    return {
        "overall": one(rows),
        "by_family": {key: one(value) for key, value in sorted(by_family.items())},
    }


def evaluate_preference_adapter(
    config: dict[str, Any],
    *,
    revision: str,
    cache_dir: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    policy_adapter: str | Path,
    reference_adapter: str | Path,
    beta: float,
    loss_type: str,
    evaluation_role: str,
) -> dict[str, Any]:
    """Score a trained adapter against its fixed parent on held-out pairs."""

    rows = read_jsonl(data_path)
    if not rows:
        raise ValueError("Preference evaluation split is empty")
    ids = [str(row.get("id") or row.get("pair_id") or "") for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Preference evaluation ids must be present and unique")

    settings = config["stage4"]["dpo_training"]
    max_length = int(settings["max_length"])
    tokenizer = load_tokenizer(config, revision, cache_dir, padding_side="right")
    model = load_generation_model(
        config, revision, cache_dir, adapter_path=policy_adapter
    )
    if "ref" in model.peft_config:
        raise RuntimeError("Policy artifact unexpectedly already contains a ref adapter")
    model.load_adapter(str(Path(reference_adapter).resolve()), adapter_name="ref", is_trainable=False)
    device = torch.device("cuda", torch.cuda.current_device())
    batch_size = int(settings.get("preference_eval_batch_size", 8))
    scored: list[dict[str, Any]] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded, completion_lengths = _encode_batch(
            tokenizer, batch, max_length=max_length
        )
        tensors = {key: value.to(device) for key, value in encoded.items()}
        model_inputs = {
            "input_ids": tensors["input_ids"],
            "attention_mask": tensors["attention_mask"],
            "use_cache": False,
        }
        model.set_adapter("default")
        with torch.inference_mode():
            policy_output = model(**model_inputs)
            policy_logps, policy_counts = _masked_sequence_logps(
                policy_output.logits,
                tensors["input_ids"],
                tensors["completion_mask"],
            )
        del policy_output
        model.set_adapter("ref")
        with torch.inference_mode():
            reference_output = model(**model_inputs)
            reference_logps, reference_counts = _masked_sequence_logps(
                reference_output.logits,
                tensors["input_ids"],
                tensors["completion_mask"],
            )
        del reference_output

        count = len(batch)
        pc, pr = policy_logps[:count], policy_logps[count:]
        rc, rr = reference_logps[:count], reference_logps[count:]
        cc, cr = policy_counts[:count], policy_counts[count:]
        if not torch.equal(policy_counts, reference_counts):
            raise RuntimeError("Policy/reference completion token counts differ")
        chosen_ratio = pc - rc
        rejected_ratio = pr - rr
        summed_margin = float(beta) * (chosen_ratio - rejected_ratio)
        normalized_margin = float(beta) * (
            chosen_ratio / cc - rejected_ratio / cr
        )
        for index, source in enumerate(batch):
            scored.append(
                {
                    "id": ids[start + index],
                    "family": source.get("family", "unknown"),
                    "difficulty": source.get("difficulty", "unknown"),
                    "pair_type": source.get("pair_type", "capability"),
                    "chosen_tokens": completion_lengths[index],
                    "rejected_tokens": completion_lengths[count + index],
                    "policy_chosen_logp": float(pc[index].item()),
                    "policy_rejected_logp": float(pr[index].item()),
                    "reference_chosen_logp": float(rc[index].item()),
                    "reference_rejected_logp": float(rr[index].item()),
                    "summed_reward_margin": float(summed_margin[index].item()),
                    "normalized_reward_margin": float(normalized_margin[index].item()),
                }
            )

    objective_margin = (
        "normalized_reward_margin" if loss_type == "sigmoid_norm" else "summed_reward_margin"
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "preference_scores.jsonl", scored)
    metrics = {
        "evaluation_input": {
            "role": evaluation_role,
            "path": str(Path(data_path).resolve()),
            "sha256": sha256_file(data_path),
            "row_id_set_sha256": hashlib.sha256(
                json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "count": len(rows),
        },
        "model_revision": revision,
        "policy_adapter": str(Path(policy_adapter).resolve()),
        "policy_adapter_sha256": sha256_directory(policy_adapter),
        "reference_adapter": str(Path(reference_adapter).resolve()),
        "reference_adapter_sha256": sha256_directory(reference_adapter),
        "beta": float(beta),
        "loss_type": loss_type,
        "objective_margin": objective_margin,
        "objective": _summarize(scored, objective_margin),
        "summed": _summarize(scored, "summed_reward_margin"),
        "normalized": _summarize(scored, "normalized_reward_margin"),
    }
    write_json(destination / "metrics.json", metrics)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics
