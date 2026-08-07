from __future__ import annotations

from pathlib import Path

from reasoning_lab.data.arithmetic import (
    build_examples,
    evaluate_expression,
    generate_dataset,
    validate_split_integrity,
)
from reasoning_lab.io_utils import read_jsonl, sha256_file


def _config() -> dict:
    return {
        "data": {
            "generator_version": "1.0.0",
            "seed": 1729,
            "train_size": 24,
            "validation_size": 8,
            "test_size": 8,
        }
    }


def _v2_config(train_size: int = 24) -> dict:
    return {
        "data": {
            "generator_version": "2.0.0",
            "seed": 1729,
            "train_size": train_size,
            "validation_size": 8,
            "test_size": 8,
            "continuation_size": 12,
        }
    }


def test_generator_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(_config(), first)
    generate_dataset(_config(), second)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        assert sha256_file(first / name) == sha256_file(second / name)


def test_v2_generator_is_deterministic_with_continuation(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = generate_dataset(_v2_config(), first)
    second_manifest = generate_dataset(_v2_config(), second)

    assert tuple(first_manifest["splits"]) == (
        "train",
        "validation",
        "test",
        "continuation",
    )
    assert first_manifest == second_manifest
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "continuation.jsonl",
    ):
        assert sha256_file(first / name) == sha256_file(second / name)


def test_v2_eval_splits_do_not_depend_on_train_size(tmp_path: Path) -> None:
    small = tmp_path / "small"
    large = tmp_path / "large"
    generate_dataset(_v2_config(train_size=24), small)
    generate_dataset(_v2_config(train_size=80), large)

    for name in ("validation.jsonl", "test.jsonl", "continuation.jsonl"):
        assert sha256_file(small / name) == sha256_file(large / name)

    small_train = read_jsonl(small / "train.jsonl")
    large_train = read_jsonl(large / "train.jsonl")
    assert large_train[: len(small_train)] == small_train


def test_splits_are_disjoint_and_labels_are_valid(tmp_path: Path) -> None:
    generate_dataset(_config(), tmp_path)
    validate_split_integrity(tmp_path)


def test_v2_four_splits_are_disjoint_and_labels_are_valid(tmp_path: Path) -> None:
    manifest = generate_dataset(_v2_config(), tmp_path)
    validate_split_integrity(tmp_path)
    assert manifest["splits"]["continuation"]["count"] == 12


def test_restricted_expression_evaluator() -> None:
    assert evaluate_expression("(-3 * 7) + 5") == -16
    assert evaluate_expression("(4 + 8) * -2") == -24
    rows = build_examples(total=20, seed=7, generator_version="test")
    assert all(evaluate_expression(row["expression"]) == row["answer"] for row in rows)


def test_prompt_templates_are_not_confounded_with_families() -> None:
    rows = build_examples(total=120, seed=11, generator_version="test")
    for family in {row["family"] for row in rows}:
        templates = {row["template_id"] for row in rows if row["family"] == family}
        assert len(templates) > 1
