from __future__ import annotations

from pathlib import Path

import pytest

from reasoning_lab.data.sft_mixture import (
    materialize_training_view,
    reasoning_completion,
)
from reasoning_lab.io_utils import read_jsonl, write_jsonl


def _rows(split: str = "train") -> list[dict]:
    expressions = [
        ("a", "2 + 3 - 4", 1),
        ("b", "-3 * 7", -21),
        ("c", "(4 * 5) + -6", 14),
        ("d", "(8 + -3) * 6", 30),
    ]
    return [
        {
            "id": item_id,
            "split": split,
            "family": "arithmetic",
            "difficulty": 1,
            "template_id": "t1",
            "expression": expression,
            "prompt": "old",
            "completion": f"FINAL_ANSWER={answer}",
            "answer": answer,
            "output_contract": "answer_only",
        }
        for item_id, expression, answer in expressions
    ]


def test_reasoning_completion_has_verified_intermediate_steps() -> None:
    assert reasoning_completion("2 + 3 - 4") == (
        "Step 1: 2 + 3 = 5.\n"
        "Step 2: 5 - 4 = 1.\n"
        "FINAL_ANSWER=1"
    )
    assert reasoning_completion("(4 + -8) * -3").endswith("FINAL_ANSWER=12")


def test_mixture_is_deterministic_and_exact(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    write_jsonl(source, _rows())
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    manifest = materialize_training_view(
        source,
        first,
        candidate_id="half",
        reasoning_fraction=0.5,
        mixture_seed=17,
    )
    materialize_training_view(
        source,
        second,
        candidate_id="half",
        reasoning_fraction=0.5,
        mixture_seed=17,
    )
    assert first.read_bytes() == second.read_bytes()
    assert manifest["output"]["style_counts"] == {
        "concise_reasoning": 2,
        "direct": 2,
    }
    rows = read_jsonl(first)
    assert all(row["source_id"] == row["id"] for row in rows)
    assert all("You may show concise arithmetic steps" in row["prompt"] for row in rows)


def test_same_ratio_has_identical_training_text_across_objectives(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    write_jsonl(source, _rows())
    nll = tmp_path / "nll.jsonl"
    dft = tmp_path / "dft.jsonl"
    materialize_training_view(
        source, nll, candidate_id="nll", reasoning_fraction=0.75, mixture_seed=9
    )
    materialize_training_view(
        source, dft, candidate_id="dft", reasoning_fraction=0.75, mixture_seed=9
    )
    nll_text = [(row["prompt"], row["completion"]) for row in read_jsonl(nll)]
    dft_text = [(row["prompt"], row["completion"]) for row in read_jsonl(dft)]
    assert nll_text == dft_text


def test_mixture_rejects_evaluation_split(tmp_path: Path) -> None:
    source = tmp_path / "validation.jsonl"
    write_jsonl(source, _rows(split="validation"))
    with pytest.raises(ValueError, match="only be derived"):
        materialize_training_view(
            source,
            tmp_path / "invalid.jsonl",
            candidate_id="invalid",
            reasoning_fraction=0.5,
            mixture_seed=1,
        )


def test_contract_conditioning_separates_reasoning_and_direct_prompts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    write_jsonl(source, _rows())
    output = tmp_path / "conditioned.jsonl"
    materialize_training_view(
        source,
        output,
        candidate_id="conditioned",
        reasoning_fraction=0.5,
        mixture_seed=17,
        prompt_conditioning="contract_conditioned",
    )
    rows = read_jsonl(output)
    reasoning = [row for row in rows if row["supervision_style"] == "concise_reasoning"]
    direct = [row for row in rows if row["supervision_style"] == "direct"]
    assert all("Show concise arithmetic steps" in row["prompt"] for row in reasoning)
    assert all(row["prompt"] == "old" for row in direct)
