from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from reasoning_lab.config import resolve_path
from reasoning_lab.data.arithmetic import generate_dataset
from reasoning_lab.data.sft_mixture import REASONING_PROMPT
from reasoning_lab.io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def _id_set_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted(str(row["id"]) for row in rows),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fewshot_prompt(question: str, demonstrations: list[dict[str, Any]]) -> str:
    blocks = [
        f"Question: {row['question']}\nAnswer: {row['source_answer']}"
        for row in demonstrations
    ]
    blocks.append(f"Question: {question.strip()}\nAnswer:")
    return "\n\n".join(blocks)


def _freeze_file(
    path: Path, rows: list[dict[str, Any]], *, role: str
) -> dict[str, Any]:
    write_jsonl(path, rows)
    return {
        "path": str(path.resolve()),
        "role": role,
        "count": len(rows),
        "sha256": sha256_file(path),
        "row_id_set_sha256": _id_set_hash(rows),
    }


def _assert_prefix(
    expanded: list[dict[str, Any]], original: list[dict[str, Any]], *, split: str
) -> None:
    if expanded[: len(original)] != original:
        raise ValueError(f"Expanded arithmetic {split} no longer preserves the frozen prefix")


def prepare_stage3_frozen_data(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    source_stage2_run: str | Path | None = None,
) -> dict[str, Any]:
    """Precommit fresh dev/confirmation sets without rewriting Stage 2 data."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Stage 3 frozen-data directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    stage3 = config["stage3"]

    original_root = resolve_path(config, "data_dir")
    original_manifest = read_json(original_root / "dataset_manifest.json")
    source_run = Path(source_stage2_run).resolve() if source_stage2_run else None
    if source_run is not None and sha256_file(
        original_root / "dataset_manifest.json"
    ) != sha256_file(source_run / "data_manifest.json"):
        raise ValueError("Arithmetic data manifest differs from the Stage 2 source run")
    for split in ("train", "validation", "test", "continuation"):
        source = original_root / f"{split}.jsonl"
        expected = str(original_manifest["splits"][split]["sha256"])
        if sha256_file(source) != expected:
            raise ValueError(f"Frozen Stage 2 arithmetic {split} hash mismatch")

    expanded_config = copy.deepcopy(config)
    expanded_config["data"]["validation_size"] = int(stage3["synthetic_dev_size"])
    expanded_config["data"]["test_size"] = int(config["data"]["test_size"]) + int(
        stage3["synthetic_confirm_size"]
    )
    expanded_root = destination / "expanded_arithmetic_v2"
    expanded_manifest = generate_dataset(expanded_config, expanded_root)
    for split in ("train", "continuation"):
        if sha256_file(expanded_root / f"{split}.jsonl") != sha256_file(
            original_root / f"{split}.jsonl"
        ):
            raise ValueError(f"Expanded arithmetic unexpectedly changed {split}")

    original_validation = read_jsonl(original_root / "validation.jsonl")
    original_test = read_jsonl(original_root / "test.jsonl")
    expanded_validation = read_jsonl(expanded_root / "validation.jsonl")
    expanded_test = read_jsonl(expanded_root / "test.jsonl")
    _assert_prefix(expanded_validation, original_validation, split="validation")
    _assert_prefix(expanded_test, original_test, split="test")
    synthetic_dev = expanded_validation[: int(stage3["synthetic_dev_size"])]
    exposed_test_ids = {str(row["id"]) for row in original_test}
    synthetic_confirm = [
        row for row in expanded_test if str(row["id"]) not in exposed_test_ids
    ][: int(stage3["synthetic_confirm_size"])]
    if len(synthetic_confirm) != int(stage3["synthetic_confirm_size"]):
        raise ValueError("Could not materialize the requested fresh synthetic confirmation set")
    if {row["id"] for row in synthetic_dev} & {row["id"] for row in synthetic_confirm}:
        raise ValueError("Synthetic validation and confirmation sets overlap")
    synthetic_contract = str(stage3.get("synthetic_contract", "answer_only"))
    if synthetic_contract == "reasoning_then_answer":
        synthetic_dev = [
            {
                **row,
                "prompt": REASONING_PROMPT.format(expression=row["expression"]),
                "output_contract": "reasoning_then_answer",
            }
            for row in synthetic_dev
        ]
        synthetic_confirm = [
            {
                **row,
                "prompt": REASONING_PROMPT.format(expression=row["expression"]),
                "output_contract": "reasoning_then_answer",
            }
            for row in synthetic_confirm
        ]
    elif synthetic_contract != "answer_only":
        raise ValueError(f"Unknown Stage 3 synthetic contract: {synthetic_contract}")

    external_root = resolve_path(config, "external_data_dir")
    external_manifest = read_json(external_root / "dataset_manifest.json")
    if source_run is not None and sha256_file(
        external_root / "dataset_manifest.json"
    ) != sha256_file(source_run / "external_data_manifest.json"):
        raise ValueError("GSM8K manifest differs from the Stage 2 source run")
    for name in (
        "source_train",
        "source_test",
        "fewshot_examples",
        "parser_validation",
        "test_eval",
    ):
        path = external_root / f"{name}.jsonl"
        if sha256_file(path) != str(external_manifest["files"][name]["sha256"]):
            raise ValueError(f"Frozen Stage 2 GSM8K {name} hash mismatch")

    source_train = read_jsonl(external_root / "source_train.jsonl")
    source_test = read_jsonl(external_root / "source_test.jsonl")
    demonstrations = read_jsonl(external_root / "fewshot_examples.jsonl")
    prior_parser = read_jsonl(external_root / "parser_validation.jsonl")
    prior_test = read_jsonl(external_root / "test_eval.jsonl")
    excluded_dev_ids = {
        str(row["id"]) for row in [*demonstrations, *prior_parser]
    }
    external_dev = [
        {**row, "prompt": _fewshot_prompt(row["question"], demonstrations),
         "output_contract": "gsm8k_flexible"}
        for row in source_train
        if str(row["id"]) not in excluded_dev_ids
    ][: int(stage3["external_dev_size"])]
    prior_test_ids = {str(row["id"]) for row in prior_test}
    external_confirm = [
        {**row, "prompt": _fewshot_prompt(row["question"], demonstrations),
         "output_contract": "gsm8k_flexible"}
        for row in source_test
        if str(row["id"]) not in prior_test_ids
    ][: int(stage3["external_confirm_size"])]
    if len(external_dev) != int(stage3["external_dev_size"]):
        raise ValueError("Could not materialize the requested GSM8K development set")
    if len(external_confirm) != int(stage3["external_confirm_size"]):
        raise ValueError("Could not materialize the requested fresh GSM8K confirmation set")
    if any(row.get("source_split") != "train" for row in external_dev):
        raise ValueError("External development data must come only from GSM8K train")
    if any(row.get("source_split") != "test" for row in external_confirm):
        raise ValueError("External confirmation data must come only from GSM8K test")
    if {row["id"] for row in external_dev} & {row["id"] for row in external_confirm}:
        raise ValueError("External development and confirmation sets overlap")

    files = {
        "synthetic_validation": _freeze_file(
            destination / "synthetic_validation.jsonl",
            synthetic_dev,
            role="synthetic_validation",
        ),
        "synthetic_confirmation": _freeze_file(
            destination / "synthetic_confirmation.jsonl",
            synthetic_confirm,
            role="synthetic_confirmation",
        ),
        "external_train_validation": _freeze_file(
            destination / "gsm8k_train_validation.jsonl",
            external_dev,
            role="external_train_validation",
        ),
        "external_confirmation": _freeze_file(
            destination / "gsm8k_confirmation.jsonl",
            external_confirm,
            role="external_confirmation",
        ),
    }
    manifest = {
        "protocol": "stage3_frozen_benchmarks_v1",
        "synthetic_contract": synthetic_contract,
        "files": files,
        "excluded_prior_holdouts": {
            "synthetic_stage2_test_count": len(original_test),
            "synthetic_stage2_test_id_set_sha256": _id_set_hash(original_test),
            "gsm8k_stage2_test_count": len(prior_test),
            "gsm8k_stage2_test_id_set_sha256": _id_set_hash(prior_test),
        },
        "source_manifests": {
            "arithmetic_v2_sha256": sha256_file(original_root / "dataset_manifest.json"),
            "gsm8k_sha256": sha256_file(external_root / "dataset_manifest.json"),
            "gsm8k_revision": external_manifest["resolved_revision"],
            "stage2_run": str(source_run) if source_run is not None else None,
        },
        "expanded_arithmetic_manifest": expanded_manifest,
    }
    write_json(destination / "frozen_data_manifest.json", manifest)
    return manifest
