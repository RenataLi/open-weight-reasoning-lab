from __future__ import annotations

import pytest

from reasoning_lab.stage3_selection import select_candidate, validate_stage3_spec


def _metrics(role: str, exact: float, *, strict: float = 0.96) -> dict:
    summary = {
        "count": 8,
        "answer_exact_match": exact,
        "format_valid_rate": 0.98,
        "strict_correct_rate": strict,
        "parse_success_rate": 1.0,
        "hit_max_new_tokens_rate": 0.0,
    }
    return {
        "overall": summary,
        "by_family": {"f": {**summary}},
        "by_difficulty": {"1": {**summary}},
        "evaluation_input": {"role": role},
    }


def _policy() -> dict:
    return {
        "max_synthetic_regression": 0.05,
        "max_external_regression": 0.05,
        "max_group_regression": 0.075,
        "min_format_valid_rate": 0.95,
        "min_parse_success_rate": 0.99,
        "max_synthetic_token_limit_rate": 0.05,
        "max_external_token_limit_rate": 0.10,
        "expected_seeds": [1, 2, 3],
    }


def _candidate(candidate_id: str, synthetic: float, external: float) -> dict:
    return {
        "id": candidate_id,
        "seeds": [
            {
                "seed": seed,
                "synthetic": _metrics("synthetic_validation", synthetic),
                "external": _metrics("external_train_validation", external),
            }
            for seed in (1, 2, 3)
        ],
    }


def test_selector_is_deterministic_and_validation_only() -> None:
    result = select_candidate(
        base_synthetic=_metrics("synthetic_validation", 0.95),
        base_external=_metrics("external_train_validation", 0.70),
        candidates=[_candidate("b", 0.94, 0.72), _candidate("a", 0.94, 0.72)],
        policy=_policy(),
    )
    assert result["selected_candidate"] == "a"
    assert result["selection_input_roles"] == [
        "external_train_validation",
        "synthetic_validation",
    ]


def test_selector_rejects_confirmation_data() -> None:
    candidate = _candidate("unsafe", 0.95, 0.70)
    candidate["seeds"][0]["synthetic"]["evaluation_input"]["role"] = (
        "synthetic_confirmation"
    )
    with pytest.raises(ValueError, match="non-validation"):
        select_candidate(
            base_synthetic=_metrics("synthetic_validation", 0.95),
            base_external=_metrics("external_train_validation", 0.70),
            candidates=[candidate],
            policy=_policy(),
        )


def test_stage3_spec_rejects_duplicate_seeds_and_invalid_loss() -> None:
    stage3 = {
        "search_seeds": [1, 1, 2],
        "candidates": [
            {
                "id": "x",
                "reasoning_fraction": 0.5,
                "loss_type": "nll",
                "learning_rate": 1e-5,
                "max_steps": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="unique seeds"):
        validate_stage3_spec(stage3)
    stage3["search_seeds"] = [1, 2, 3]
    stage3["candidates"][0]["loss_type"] = "unknown"
    with pytest.raises(ValueError, match="loss_type"):
        validate_stage3_spec(stage3)
