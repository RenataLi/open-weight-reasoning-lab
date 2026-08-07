from __future__ import annotations

import pytest

from reasoning_lab.stage4_selection import select_dpo_candidate


def _generation(role: str, exact: float = 0.7, length: float = 20.0) -> dict:
    group = {
        "answer_exact_match": exact,
        "format_valid_rate": 1.0,
        "strict_correct_rate": exact,
        "parse_success_rate": 1.0,
        "hit_max_new_tokens_rate": 0.0,
        "generated_tokens_mean": length,
    }
    return {
        "evaluation_input": {"role": role},
        "overall": dict(group),
        "by_family": {"f": dict(group)},
    }


def _preference(role: str = "stage4_preference_validation") -> dict:
    return {
        "evaluation_input": {"role": role},
        "objective": {
            "overall": {"reward_accuracy": 0.7, "reward_margin_mean": 0.2}
        },
    }


def _policy() -> dict:
    return {
        "max_synthetic_regression": 0.02,
        "max_external_regression": 0.03,
        "max_family_regression": 0.05,
        "min_preference_accuracy": 0.6,
        "min_synthetic_format_valid_rate": 0.95,
        "min_external_format_valid_rate": 0.15,
        "max_synthetic_format_regression": 0.03,
        "max_external_format_regression": 0.03,
        "max_synthetic_strict_regression": 0.05,
        "max_external_strict_regression": 0.05,
        "min_parse_success_rate": 0.99,
        "max_synthetic_token_limit_rate": 0.05,
        "max_external_token_limit_rate": 0.1,
        "min_length_ratio": 0.85,
        "max_length_ratio": 1.15,
    }


def test_stage4_selector_uses_validation_only_and_is_deterministic() -> None:
    base_s = _generation("stage4_synthetic_validation")
    base_e = _generation("stage4_svamp_validation", exact=0.5)
    candidates = [
        {
            "id": "b",
            "synthetic": _generation("stage4_synthetic_validation", exact=0.71),
            "external": _generation("stage4_svamp_validation", exact=0.51),
            "preference": _preference(),
        },
        {
            "id": "a",
            "synthetic": _generation("stage4_synthetic_validation", exact=0.71),
            "external": _generation("stage4_svamp_validation", exact=0.51),
            "preference": _preference(),
        },
    ]
    result = select_dpo_candidate(
        base_synthetic=base_s,
        base_external=base_e,
        candidates=candidates,
        expected_candidate_ids=["a", "b"],
        policy=_policy(),
    )
    assert result["selected_candidate"] == "a"

    candidates[0]["preference"] = _preference("stage4_preference_confirmation")
    with pytest.raises(ValueError, match="non-validation preference"):
        select_dpo_candidate(
            base_synthetic=base_s,
            base_external=base_e,
            candidates=candidates,
            expected_candidate_ids=["a", "b"],
            policy=_policy(),
        )

    candidates[0]["preference"] = _preference()
    candidates[0]["external"] = _generation(
        "stage4_svamp_development_quarantine"
    )
    with pytest.raises(ValueError, match="non-validation SVAMP"):
        select_dpo_candidate(
            base_synthetic=base_s,
            base_external=base_e,
            candidates=candidates,
            expected_candidate_ids=["a", "b"],
            policy=_policy(),
        )


def test_stage4_selector_rejects_regression_or_weak_preference() -> None:
    candidate = {
        "id": "x",
        "synthetic": _generation("stage4_synthetic_validation", exact=0.60),
        "external": _generation("stage4_svamp_validation", exact=0.50),
        "preference": _preference(),
    }
    candidate["preference"]["objective"]["overall"]["reward_accuracy"] = 0.55
    result = select_dpo_candidate(
        base_synthetic=_generation("stage4_synthetic_validation", exact=0.70),
        base_external=_generation("stage4_svamp_validation", exact=0.50),
        candidates=[candidate],
        expected_candidate_ids=["x"],
        policy=_policy(),
    )
    assert result["selected_candidate"] is None
    checks = result["records"][0]["checks"]
    assert not checks["synthetic_retention"]
    assert not checks["preference_accuracy"]


def test_stage4_selector_uses_relative_format_retention_with_floor() -> None:
    base_s = _generation("stage4_synthetic_validation")
    base_s["overall"]["format_valid_rate"] = 0.83
    candidate_s = _generation("stage4_synthetic_validation")
    candidate_s["overall"]["format_valid_rate"] = 0.79
    candidate = {
        "id": "x",
        "synthetic": candidate_s,
        "external": _generation("stage4_svamp_validation"),
        "preference": _preference(),
    }
    policy = _policy()
    policy["min_synthetic_format_valid_rate"] = 0.75
    result = select_dpo_candidate(
        base_synthetic=base_s,
        base_external=_generation("stage4_svamp_validation"),
        candidates=[candidate],
        expected_candidate_ids=["x"],
        policy=policy,
    )
    assert not result["records"][0]["checks"]["synthetic_format"]

    candidate_s["overall"]["format_valid_rate"] = 0.80
    result = select_dpo_candidate(
        base_synthetic=base_s,
        base_external=_generation("stage4_svamp_validation"),
        candidates=[candidate],
        expected_candidate_ids=["x"],
        policy=policy,
    )
    assert result["records"][0]["checks"]["synthetic_format"]
