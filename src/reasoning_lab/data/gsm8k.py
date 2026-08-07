from __future__ import annotations

import hashlib
import importlib.metadata
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
from huggingface_hub import snapshot_download

from reasoning_lab.evaluation.parsing import normalize_number
from reasoning_lab.io_utils import sha256_file, write_json, write_jsonl


FINAL_ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)\s*$")
CALCULATION_ANNOTATION_PATTERN = re.compile(r"<<.*?>>")
SOURCE_URL = "https://huggingface.co/datasets/openai/gsm8k"


def extract_gsm8k_answer(solution: str) -> str:
    match = FINAL_ANSWER_PATTERN.search(solution)
    if match is None:
        raise ValueError("GSM8K solution does not contain a final #### answer")
    return normalize_number(match.group(1).strip())


def _clean_reasoning(solution: str) -> str:
    reasoning = solution.rsplit("####", maxsplit=1)[0].strip()
    reasoning = CALCULATION_ANNOTATION_PATTERN.sub("", reasoning)
    return reasoning.strip()


def _prompt(question: str, demonstrations: list[dict[str, Any]]) -> str:
    blocks = [
        f"Question: {row['question']}\nAnswer: {row['source_answer']}" for row in demonstrations
    ]
    blocks.append(f"Question: {question.strip()}\nAnswer:")
    return "\n\n".join(blocks)


def _normalize_row(row: dict[str, Any], source_split: str, source_index: int) -> dict[str, Any]:
    question = str(row["question"]).strip()
    source_answer = str(row["answer"])
    answer = extract_gsm8k_answer(source_answer)
    reasoning = _clean_reasoning(source_answer)
    stable_key = f"gsm8k|{question}"
    example_id = "gsm8k_" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:20]
    return {
        "id": example_id,
        "family": "gsm8k",
        "difficulty": "unknown",
        "template_id": "gsm8k_reasoning_v1",
        "question": question,
        "prompt": None,
        "completion": f"{reasoning}\nFINAL_ANSWER={answer}",
        "answer": answer,
        "output_contract": "reasoning_then_answer",
        "source_dataset": "openai/gsm8k",
        "source_split": source_split,
        "source_index": source_index,
        "source_answer": source_answer,
    }


def _hash_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}|{row['id']}".encode("utf-8")).hexdigest(),
    )


def prepare_gsm8k_snapshot(
    config: dict[str, Any],
    *,
    resolved_revision: str,
    cache_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    external = config["external_evaluation"]
    dataset_id = str(external["dataset_id"])
    subset = str(external["subset"])
    source_root = Path(output_dir) / "source" / resolved_revision
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=resolved_revision,
        cache_dir=str(cache_dir),
        local_dir=str(source_root),
        allow_patterns=[".gitattributes", "README.md", "LICENSE", "eval.yaml", "main/*.parquet"],
    )
    train_parquet = source_root / "main" / "train-00000-of-00001.parquet"
    test_parquet = source_root / "main" / "test-00000-of-00001.parquet"
    expected_files = external.get("expected_files", {})
    for name, path in (("train", train_parquet), ("test", test_parquet)):
        if not path.exists():
            raise FileNotFoundError(f"Missing GSM8K source parquet: {path}")
        expected = expected_files.get(name)
        if expected:
            if path.stat().st_size != int(expected["bytes"]):
                raise ValueError(f"GSM8K {name} parquet size mismatch")
            if sha256_file(path) != str(expected["sha256"]):
                raise ValueError(f"GSM8K {name} parquet checksum mismatch")

    source_train = parquet.read_table(train_parquet).to_pylist()
    source_test = parquet.read_table(test_parquet).to_pylist()
    train_rows = [
        _normalize_row(dict(row), "train", index) for index, row in enumerate(source_train)
    ]
    test_rows = [
        _normalize_row(dict(row), "test", index) for index, row in enumerate(source_test)
    ]
    train_rows = _hash_order(train_rows, int(external["subset_seed"]))
    test_rows = _hash_order(test_rows, int(external["subset_seed"]))

    fewshot_count = int(external["fewshot_count"])
    validation_size = int(external["parser_validation_size"])
    test_size = int(external["test_size"])
    if fewshot_count + validation_size > len(train_rows):
        raise ValueError("Requested GSM8K train-derived evaluation subsets exceed the source split")
    if test_size > len(test_rows):
        raise ValueError("Requested GSM8K test subset exceeds the source test split")

    demonstrations = train_rows[:fewshot_count]
    parser_validation = train_rows[fewshot_count : fewshot_count + validation_size]
    test_eval = test_rows[:test_size]
    for row in parser_validation:
        row["prompt"] = _prompt(row["question"], demonstrations)
        row["output_contract"] = "gsm8k_flexible"
    for row in test_eval:
        row["prompt"] = _prompt(row["question"], demonstrations)
        row["output_contract"] = "gsm8k_flexible"

    subsets = {
        "source_train": train_rows,
        "source_test": test_rows,
        "fewshot_examples": demonstrations,
        "parser_validation": parser_validation,
        "test_eval": test_eval,
    }
    derived_ids = [
        {row["id"] for row in subsets[name]}
        for name in ("fewshot_examples", "parser_validation", "test_eval")
    ]
    for index, left in enumerate(derived_ids):
        for right in derived_ids[index + 1 :]:
            if left & right:
                raise ValueError("GSM8K derived subsets overlap")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for name, rows in subsets.items():
        path = destination / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {"count": len(rows), "sha256": sha256_file(path)}

    manifest = {
        "dataset_id": dataset_id,
        "subset": subset,
        "requested_revision": str(external["revision"]),
        "resolved_revision": resolved_revision,
        "source_url": SOURCE_URL,
        "license": "MIT",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "library_versions": {
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        },
        "subset_seed": int(external["subset_seed"]),
        "source_files": {
            "train": {
                "path": str(train_parquet),
                "bytes": train_parquet.stat().st_size,
                "sha256": sha256_file(train_parquet),
            },
            "test": {
                "path": str(test_parquet),
                "bytes": test_parquet.stat().st_size,
                "sha256": sha256_file(test_parquet),
            },
        },
        "fewshot_ids": [row["id"] for row in demonstrations],
        "files": files,
    }
    write_json(destination / "dataset_manifest.json", manifest)
    return manifest
