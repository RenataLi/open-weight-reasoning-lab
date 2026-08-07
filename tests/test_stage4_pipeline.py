from __future__ import annotations

from reasoning_lab.stage4_pipeline import _confirmation_verdict


def _comparison(delta: float, low: float, worst: float | None = None) -> dict:
    return {
        "answer_exact_match_delta": delta,
        "paired_bootstrap_95_ci": [low, max(low, delta + 0.02)],
        "worst_family_delta": delta if worst is None else worst,
    }


def _fixture() -> tuple[dict, dict, dict]:
    config = {
        "stage4": {
            "primary_benefit_endpoint": "synthetic",
            "confirmation_policy": {
                "max_synthetic_regression": 0.02,
                "max_synthetic_ci_regression": 0.05,
                "max_family_regression": 0.05,
                "min_preference_accuracy": 0.60,
                "min_preference_accuracy_advantage_vs_control": 0.02,
                "min_synthetic_format_valid_rate": 0.95,
                "max_synthetic_format_regression": 0.03,
                "max_synthetic_strict_regression": 0.03,
                "min_parse_success_rate": 0.99,
                "max_synthetic_token_limit_rate": 0.05,
                "min_length_ratio": 0.85,
                "max_length_ratio": 1.15,
                "objective_benefit_delta": 0.02,
            }
        }
    }
    overall = {
        "answer_exact_match": 0.7,
        "format_valid_rate": 1.0,
        "strict_correct_rate": 0.7,
        "parse_success_rate": 1.0,
        "hit_max_new_tokens_rate": 0.0,
        "generated_tokens_mean": 20.0,
    }
    metrics = {
        "parent": {
            "synthetic": {"overall": dict(overall)},
        },
        "dpo": {
            "synthetic": {"overall": dict(overall)},
        },
        "dpo_preference": {
            "objective": {
                "overall": {"reward_accuracy": 0.75, "reward_margin_mean": 0.2}
            }
        },
        "control_preference": {
            "objective": {
                "overall": {"reward_accuracy": 0.70, "reward_margin_mean": 0.1}
            }
        },
    }
    comparisons = {
        "parent_to_dpo": {
            "synthetic": _comparison(0.0, -0.01),
        },
        "control_to_dpo": {
            "synthetic": _comparison(0.03, 0.01),
        },
    }
    return config, metrics, comparisons


def test_confirmation_verdict_distinguishes_benefit_inconclusive_and_failure() -> None:
    config, metrics, comparisons = _fixture()
    result = _confirmation_verdict(config, metrics=metrics, comparisons=comparisons)
    assert result["status"] == "OBJECTIVE-SPECIFIC BENEFIT SUPPORTED"

    comparisons["control_to_dpo"]["synthetic"] = _comparison(0.0, -0.01)
    result = _confirmation_verdict(config, metrics=metrics, comparisons=comparisons)
    assert result["status"] == "DPO PIPELINE VALIDATED; BENEFIT INCONCLUSIVE"

    comparisons["control_to_dpo"]["synthetic"] = _comparison(0.01, -0.01)
    result = _confirmation_verdict(config, metrics=metrics, comparisons=comparisons)
    assert result["status"] == "DPO PIPELINE VALIDATED; BENEFIT INCONCLUSIVE"

    metrics["dpo_preference"]["objective"]["overall"]["reward_accuracy"] = 0.5
    result = _confirmation_verdict(config, metrics=metrics, comparisons=comparisons)
    assert result["status"] == "DPO PILOT FAILED SAFETY GATE"
