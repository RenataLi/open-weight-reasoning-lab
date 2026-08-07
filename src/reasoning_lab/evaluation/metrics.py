from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty prediction set")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
        by_difficulty[str(row["difficulty"])].append(row)

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        token_counts = [int(item["generated_tokens"]) for item in items]
        prompt_counts = [
            int(item["prompt_tokens"]) for item in items if "prompt_tokens" in item
        ]
        result = {
            "count": len(items),
            "answer_exact_match": _safe_mean([float(item["answer_exact_match"]) for item in items]),
            "format_valid_rate": _safe_mean([float(item["format_valid"]) for item in items]),
            "strict_correct_rate": _safe_mean([float(item["strict_correct"]) for item in items]),
            "parse_success_rate": _safe_mean([float(item["parse_success"]) for item in items]),
            "hit_max_new_tokens_rate": _safe_mean(
                [float(item["hit_max_new_tokens"]) for item in items]
            ),
            "generated_tokens_mean": _safe_mean([float(value) for value in token_counts]),
            "generated_tokens_median": float(statistics.median(token_counts)),
        }
        if prompt_counts:
            result["prompt_tokens_mean"] = _safe_mean(
                [float(value) for value in prompt_counts]
            )
            result["prompt_tokens_max"] = max(prompt_counts)
        return result

    return {
        "overall": summary(rows),
        "by_family": {key: summary(value) for key, value in sorted(by_family.items())},
        "by_difficulty": {key: summary(value) for key, value in sorted(by_difficulty.items())},
    }


def compare_predictions(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len({row["id"] for row in baseline}) != len(baseline):
        raise ValueError("Baseline predictions contain duplicate IDs")
    if len({row["id"] for row in candidate}) != len(candidate):
        raise ValueError("Candidate predictions contain duplicate IDs")
    base_by_id = {row["id"]: row for row in baseline}
    candidate_by_id = {row["id"]: row for row in candidate}
    if set(base_by_id) != set(candidate_by_id):
        raise ValueError("Baseline and candidate prediction IDs differ")

    ids = sorted(base_by_id)
    for item_id in ids:
        for field in (
            "expected_answer",
            "family",
            "difficulty",
            "task_input",
            "prompt",
            "output_contract",
        ):
            if base_by_id[item_id].get(field) != candidate_by_id[item_id].get(field):
                raise ValueError(
                    f"Paired prediction metadata differs for {item_id!r}: {field}"
                )
    deltas = [
        float(candidate_by_id[item_id]["answer_exact_match"])
        - float(base_by_id[item_id]["answer_exact_match"])
        for item_id in ids
    ]
    rng = random.Random(bootstrap_seed)
    bootstrapped: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        bootstrapped.append(_safe_mean(sample))
    bootstrapped.sort()

    def percentile(values: list[float], probability: float) -> float:
        index = min(len(values) - 1, max(0, round(probability * (len(values) - 1))))
        return values[index]

    per_family: dict[str, float] = {}
    families = sorted({str(row["family"]) for row in baseline})
    for family in families:
        family_ids = [item_id for item_id in ids if str(base_by_id[item_id]["family"]) == family]
        per_family[family] = _safe_mean(
            [
                float(candidate_by_id[item_id]["answer_exact_match"])
                - float(base_by_id[item_id]["answer_exact_match"])
                for item_id in family_ids
            ]
        )

    wrong_to_right = sum(
        not bool(base_by_id[item_id]["answer_exact_match"])
        and bool(candidate_by_id[item_id]["answer_exact_match"])
        for item_id in ids
    )
    right_to_wrong = sum(
        bool(base_by_id[item_id]["answer_exact_match"])
        and not bool(candidate_by_id[item_id]["answer_exact_match"])
        for item_id in ids
    )
    return {
        "count": len(ids),
        "answer_exact_match_delta": _safe_mean(deltas),
        "paired_bootstrap_95_ci": [
            percentile(bootstrapped, 0.025),
            percentile(bootstrapped, 0.975),
        ],
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "per_family_delta": per_family,
        "worst_family_delta": min(per_family.values()) if per_family else 0.0,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
