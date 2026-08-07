from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from reasoning_lab.evaluation.metrics import aggregate_predictions
from reasoning_lab.evaluation.parsing import (
    normalize_number,
    parse_arithmetic_answer,
    parse_gsm8k_answer,
)
from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from reasoning_lab.models.loading import load_generation_model, load_tokenizer


def _generated_length(token_ids: list[int], eos_ids: set[int], pad_id: int | None) -> int:
    length = 0
    for token_id in token_ids:
        if token_id in eos_ids or (pad_id is not None and token_id == pad_id):
            break
        length += 1
    return length


def evaluate_model(
    config: dict[str, Any],
    *,
    revision: str,
    cache_dir: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    adapter_path: str | Path | None = None,
    evaluation_role: str = "unspecified",
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    rows = read_jsonl(data_path)[: int(evaluation["max_examples"])]
    if not rows:
        raise ValueError("Evaluation split is empty")

    tokenizer = load_tokenizer(
        config,
        revision,
        cache_dir,
        padding_side="left",
    )
    max_prompt_tokens = int(
        evaluation.get("max_prompt_tokens", config["training"]["max_length"])
    )
    prompt_token_counts = [
        len(tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]) for row in rows
    ]
    longest_prompt = max(prompt_token_counts)
    if longest_prompt > max_prompt_tokens:
        raise ValueError(
            f"Evaluation prompt has {longest_prompt} tokens, exceeding the configured "
            f"max_prompt_tokens={max_prompt_tokens}; refusing silent truncation"
        )
    model = load_generation_model(config, revision, cache_dir, adapter_path)
    device = torch.device("cuda", torch.cuda.current_device())
    batch_size = int(evaluation["batch_size"])
    stop_strings = [str(item) for item in evaluation.get("stop_strings", [])]
    predictions: list[dict[str, Any]] = []
    eos_ids = (
        {int(tokenizer.eos_token_id)}
        if isinstance(tokenizer.eos_token_id, int)
        else {int(item) for item in (tokenizer.eos_token_id or [])}
    )

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        input_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {}
            if stop_strings:
                generation_kwargs.update(
                    {"stop_strings": stop_strings, "tokenizer": tokenizer}
                )
            outputs = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(evaluation["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                **generation_kwargs,
            )
        generated = outputs[:, input_width:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        batch_prompt_lengths = prompt_token_counts[start : start + len(batch)]
        for source, raw_text, token_ids, prompt_tokens in zip(
            batch,
            decoded,
            generated.tolist(),
            batch_prompt_lengths,
            strict=True,
        ):
            output_contract = str(source.get("output_contract", "answer_only"))
            if output_contract == "gsm8k_flexible":
                parsed = parse_gsm8k_answer(raw_text)
            elif output_contract in {"answer_only", "reasoning_then_answer"}:
                parsed = parse_arithmetic_answer(raw_text)
            else:
                raise ValueError(f"Unknown output contract: {output_contract}")
            expected_answer = normalize_number(source["answer"])
            exact = parsed.parse_success and parsed.value == expected_answer
            if output_contract == "answer_only":
                format_valid = parsed.answer_only_format
            elif output_contract == "reasoning_then_answer":
                format_valid = parsed.final_line_format
            elif output_contract == "gsm8k_flexible":
                format_valid = parsed.final_line_format
            generated_tokens = _generated_length(
                token_ids,
                eos_ids,
                tokenizer.pad_token_id,
            )
            predictions.append(
                {
                    "id": source["id"],
                    "family": source["family"],
                    "difficulty": source["difficulty"],
                    "template_id": source["template_id"],
                    "task_input": source.get("expression", source.get("question")),
                    "prompt": source["prompt"],
                    "expected_answer": expected_answer,
                    "raw_output": raw_text,
                    "parsed_answer": parsed.value,
                    "parse_success": parsed.parse_success,
                    "output_contract": output_contract,
                    "format_valid": format_valid,
                    "strict_correct": format_valid and exact,
                    "answer_exact_match": exact,
                    "generated_tokens": generated_tokens,
                    "prompt_tokens": prompt_tokens,
                    "hit_max_new_tokens": generated_tokens >= int(evaluation["max_new_tokens"]),
                }
            )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "predictions.jsonl", predictions)
    metrics = aggregate_predictions(predictions)
    metrics["model_revision"] = revision
    metrics["adapter_path"] = str(adapter_path) if adapter_path else None
    metrics["evaluation_input"] = {
        "role": evaluation_role,
        "path": str(Path(data_path).resolve()),
        "sha256": sha256_file(data_path),
        "row_id_set_sha256": hashlib.sha256(
            json.dumps(
                sorted(str(row["id"]) for row in rows),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "count": len(rows),
    }
    metrics["generation"] = {
        "do_sample": False,
        "load_in_4bit": bool(
            evaluation.get("load_in_4bit", config["quantization"]["load_in_4bit"])
        ),
        "max_new_tokens": int(evaluation["max_new_tokens"]),
        "batch_size": batch_size,
        "max_prompt_tokens": max_prompt_tokens,
        "longest_prompt_tokens": longest_prompt,
        "stop_strings": stop_strings,
    }
    write_json(destination / "metrics.json", metrics)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics
