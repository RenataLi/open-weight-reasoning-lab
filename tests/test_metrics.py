import pytest

from reasoning_lab.evaluation.metrics import aggregate_predictions, compare_predictions


def _row(item_id: str, family: str, correct: bool) -> dict:
    return {
        "id": item_id,
        "family": family,
        "difficulty": 1,
        "expected_answer": "1",
        "task_input": "1 + 0",
        "answer_exact_match": correct,
        "format_valid": True,
        "strict_correct": correct,
        "parse_success": True,
        "hit_max_new_tokens": False,
        "generated_tokens": 3,
    }


def test_aggregate_and_paired_comparison() -> None:
    baseline = [_row("a", "add", False), _row("b", "mul", True)]
    candidate = [_row("a", "add", True), _row("b", "mul", True)]
    metrics = aggregate_predictions(candidate)
    assert metrics["overall"]["answer_exact_match"] == 1.0
    comparison = compare_predictions(
        baseline,
        candidate,
        bootstrap_samples=100,
        bootstrap_seed=9,
    )
    assert comparison["answer_exact_match_delta"] == 0.5
    assert comparison["wrong_to_right"] == 1
    assert comparison["right_to_wrong"] == 0
    assert comparison["worst_family_delta"] == 0.0


def test_duplicate_prediction_ids_are_rejected() -> None:
    duplicate = [_row("a", "add", False), _row("a", "add", True)]
    with pytest.raises(ValueError, match="duplicate IDs"):
        compare_predictions(duplicate, [_row("a", "add", True)], bootstrap_samples=10, bootstrap_seed=1)


def test_paired_answer_mismatch_is_rejected() -> None:
    baseline = [_row("a", "add", False)]
    candidate = [_row("a", "add", True)]
    candidate[0]["expected_answer"] = "2"
    with pytest.raises(ValueError, match="metadata differs"):
        compare_predictions(baseline, candidate, bootstrap_samples=10, bootstrap_seed=1)
