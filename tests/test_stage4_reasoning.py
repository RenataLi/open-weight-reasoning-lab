from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from reasoning_lab.data.stage4_reasoning import (
    FAMILIES,
    ROLES,
    build_stage4_roles,
    generate_stage4_dataset,
    validate_stage4_dataset,
    validate_stage4_roles,
    validate_stage4_row,
)
from reasoning_lab.io_utils import read_jsonl, sha256_file
from reasoning_lab.data.stage4_retired_v1 import retired_v1_semantic_maybe_contains


def _sizes(size: int = 8) -> dict[str, int]:
    return {role: size for role in ROLES}


def _all_rows(rows_by_role: dict[str, list[dict]]) -> list[dict]:
    return [row for role in ROLES for row in rows_by_role[role]]


def test_stage4_roles_are_deterministic_disjoint_and_multi_domain() -> None:
    first = build_stage4_roles(sizes=_sizes(), seed=4704)
    second = build_stage4_roles(sizes=_sizes(), seed=4704)

    assert first == second
    integrity = validate_stage4_roles(first)
    rows = _all_rows(first)
    assert integrity["total_rows"] == 40
    assert integrity["unique_ids"] == 40
    assert integrity["unique_task_hashes"] == 40
    assert integrity["unique_semantic_hashes"] == 40
    assert integrity["unique_prompt_hashes"] == 40
    assert integrity["unique_normalized_prompt_hashes"] == 40
    assert integrity["overlap_count"] == 0
    directions = integrity["rejected_error_diagnostics"]["directions"]
    assert directions["higher"] + directions["lower"] == 40
    assert {row["family"] for row in rows} == set(FAMILIES)
    assert {row["domain"] for row in rows} == {
        "mathematical",
        "logical",
        "structured",
    }
    assert {row["difficulty"] for row in rows} == {1, 2, 3}
    assert len({row["prompt"] for row in rows}) == len(rows)
    assert len({row["semantic_sha256"] for row in rows}) == len(rows)
    for role in ROLES:
        directions = [row["error_direction"] for row in first[role]]
        assert abs(directions.count("higher") - directions.count("lower")) <= len(
            FAMILIES
        )
        for family in FAMILIES:
            family_directions = [
                row["error_direction"]
                for row in first[role]
                if row["family"] == family
            ]
            assert abs(
                family_directions.count("higher")
                - family_directions.count("lower")
            ) <= 1


def test_rows_match_trl_and_generation_evaluation_schemas() -> None:
    rows_by_role = build_stage4_roles(sizes=_sizes(3), seed=99)
    preference_keys = {"prompt", "chosen", "rejected"}
    benchmark_keys = {
        "id",
        "family",
        "difficulty",
        "template_id",
        "prompt",
        "answer",
        "output_contract",
    }

    for role in (
        "preference_train",
        "preference_validation",
        "preference_confirmation",
    ):
        for row in rows_by_role[role]:
            assert preference_keys <= row.keys()
            assert all(isinstance(row[key], str) for key in preference_keys)
    for role in ("benchmark_validation", "benchmark_confirmation"):
        for row in rows_by_role[role]:
            assert benchmark_keys <= row.keys()
            assert isinstance(row["answer"], int)
            assert row["output_contract"] == "reasoning_then_answer"


def test_rejected_trace_has_one_local_error_propagated_to_final() -> None:
    rows = _all_rows(build_stage4_roles(sizes=_sizes(4), seed=2048))
    final_pattern = re.compile(r"FINAL_ANSWER=(-?\d+)\Z")

    for row in rows:
        validate_stage4_row(row)
        marked = [step["name"] for step in row["rejected_steps"] if step["corrupted"]]
        assert marked == [row["corrupted_step"]]
        assert row["corrupted_step"] != row["rejected_steps"][-1]["name"]
        chosen_answer = int(final_pattern.search(row["chosen"]).group(1))
        rejected_answer = int(final_pattern.search(row["rejected"]).group(1))
        assert chosen_answer == row["answer"]
        assert rejected_answer != row["answer"]
        assert len(row["canonical_steps"]) == len(row["rejected_steps"])
        assert row["pair_length"]["character_delta"] == abs(
            len(row["chosen"]) - len(row["rejected"])
        )
        local_index = row["error_step_index"]
        local_delta = (
            row["rejected_steps"][local_index]["value"]
            - row["canonical_steps"][local_index]["value"]
        )
        final_delta = rejected_answer - chosen_answer
        assert row["local_error_delta"] == local_delta
        assert row["final_error_delta"] == final_delta
        assert 0 < abs(local_delta) <= row["local_error_bound"] <= 256
        assert row["error_direction"] == ("higher" if final_delta > 0 else "lower")
        assert row["corrupted_step"] in row["eligible_corrupted_steps"]
        if row["family"] == "ratio_allocation" and row["corrupted_step"] == "s3":
            assert row["rejected_steps"][local_index]["value"] > 0


def test_semantic_invariants_and_error_positions() -> None:
    rows = _all_rows(build_stage4_roles(sizes=_sizes(20), seed=20260807))

    logic_rows = [row for row in rows if row["family"] == "logic_code"]
    assert logic_rows
    assert all("100*b1 + 10*b2 + b3" in row["prompt"] for row in logic_rows)
    table_rows = [row for row in rows if row["family"] == "table_aggregation"]
    assert table_rows
    assert all(row["params"]["adjustment"] != row["params"]["penalty"] for row in table_rows)

    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        eligible = {
            step for row in family_rows for step in row["eligible_corrupted_steps"]
        }
        observed = {row["corrupted_step"] for row in family_rows}
        if len(eligible) >= 2 and len(family_rows) >= 12:
            assert len(observed) >= 2
            counts = {
                step: sum(row["corrupted_step"] == step for row in family_rows)
                for step in observed
            }
            shares = [count / len(family_rows) for count in counts.values()]
            assert min(shares) >= 0.15
            assert max(shares) <= 0.80


@pytest.mark.parametrize("mutation", ("answer", "prompt", "chosen", "propagation", "identity"))
def test_strong_verifier_rejects_tampering(mutation: str) -> None:
    row = copy.deepcopy(
        build_stage4_roles(sizes=_sizes(1), seed=4704)["preference_train"][0]
    )
    if mutation == "answer":
        row["answer"] += 1
    elif mutation == "prompt":
        row["prompt"] += " altered"
    elif mutation == "chosen":
        row["chosen"] = row["chosen"].replace("FINAL_ANSWER=", "FINAL_ANSWER=9", 1)
    elif mutation == "propagation":
        row["rejected_steps"][-1]["value"] += 1
    elif mutation == "identity":
        row["params"] = copy.deepcopy(row["params"])
        first_integer_key = next(
            key for key, value in row["params"].items() if isinstance(value, int)
        )
        row["params"][first_integer_key] += 1
    else:  # pragma: no cover - the parameterization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        validate_stage4_row(row)


def test_hash_partitions_preserve_role_prefixes_when_one_quota_grows() -> None:
    small_sizes = _sizes(5)
    expanded_sizes = {**small_sizes, "preference_train": 17}
    small = build_stage4_roles(sizes=small_sizes, seed=1729)
    expanded = build_stage4_roles(sizes=expanded_sizes, seed=1729)

    assert expanded["preference_train"][:5] == small["preference_train"]
    for role in ROLES[1:]:
        assert expanded[role] == small[role]


def test_reserved_prompts_are_excluded_after_normalization() -> None:
    baseline = build_stage4_roles(sizes=_sizes(2), seed=8080)
    reserved_prompt = baseline["benchmark_confirmation"][0]["prompt"]
    reserved_variant = "  \n".join(reserved_prompt.upper().split())

    rebuilt = build_stage4_roles(
        sizes=_sizes(2),
        seed=8080,
        reserved_prompts={reserved_variant},
    )
    prompts = {" ".join(row["prompt"].casefold().split()) for row in _all_rows(rebuilt)}
    assert " ".join(reserved_prompt.casefold().split()) not in prompts


def test_retired_v1_semantic_filter_contains_known_historical_task() -> None:
    assert retired_v1_semantic_maybe_contains(
        "54b094c765b108d6e1010b391b4c5b5ddfd9cbeabbbdc53a5d78ceec5957c8e0"
    )
    assert not retired_v1_semantic_maybe_contains("0" * 64)


def test_files_and_manifest_are_reproducible_and_self_verifying(tmp_path: Path) -> None:
    config = {
        "generator_version": "stage4_reasoning_test_v1",
        "seed": 73,
        **{f"{role}_size": 5 for role in ROLES},
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = generate_stage4_dataset(config, first)
    second_manifest = generate_stage4_dataset(config, second)

    assert first_manifest == second_manifest
    assert sha256_file(first / "dataset_manifest.json") == sha256_file(
        second / "dataset_manifest.json"
    )
    assert first_manifest["integrity"]["overlap_count"] == 0
    assert first_manifest["integrity"]["total_rows"] == 25
    for role in ROLES:
        assert sha256_file(first / f"{role}.jsonl") == sha256_file(
            second / f"{role}.jsonl"
        )
        assert first_manifest["roles"][role]["row_id_set_sha256"] == second_manifest[
            "roles"
        ][role]["row_id_set_sha256"]
        assert len(read_jsonl(first / f"{role}.jsonl")) == 5
    assert validate_stage4_dataset(first)["total_rows"] == 25


def test_optional_tokenizer_length_policy_is_recorded_and_enforced(tmp_path: Path) -> None:
    class WhitespaceTokenizer:
        eos_token = " <eos>"

        def __call__(self, text: str, **_: object) -> dict[str, list[str]]:
            return {"input_ids": text.split()}

    whitespace_tokenizer = WhitespaceTokenizer()

    config = {
        "seed": 101,
        **{f"{role}_size": 3 for role in ROLES},
        "max_pair_token_delta": 2,
        "max_completion_tokens": 100,
    }
    manifest = generate_stage4_dataset(config, tmp_path, tokenizer=whitespace_tokenizer)
    assert manifest["length_policy"]["tokenizer_checked"] is True
    assert manifest["integrity"]["token_lengths"]["maximum_pair_delta"] <= 2
    validate_stage4_dataset(
        tmp_path,
        tokenizer=whitespace_tokenizer,
        max_pair_token_delta=2,
        max_completion_tokens=100,
    )

    row = read_jsonl(tmp_path / "preference_train.jsonl")[0]

    with pytest.raises(ValueError, match="token-length delta"):
        validate_stage4_row(
            row,
            tokenizer=whitespace_tokenizer,
            max_pair_token_delta=-1,
        )


def test_invalid_role_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly"):
        build_stage4_roles(sizes={"preference_train": 1})
    with pytest.raises(ValueError, match="non-negative"):
        build_stage4_roles(sizes={**_sizes(1), "benchmark_confirmation": -1})
