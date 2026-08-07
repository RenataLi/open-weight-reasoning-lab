from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any


ALLOWED_LOSSES = {"nll", "dft"}
VALIDATION_ROLES = {"synthetic_validation", "external_train_validation"}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_stage3_spec(stage3: dict[str, Any]) -> None:
    if str(stage3.get("synthetic_contract", "answer_only")) not in {
        "answer_only",
        "reasoning_then_answer",
    }:
        raise ValueError("Unknown stage3.synthetic_contract")
    seeds = [int(seed) for seed in stage3.get("search_seeds", [])]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("stage3.search_seeds must contain at least three unique seeds")
    candidates = stage3.get("candidates", [])
    ids = [str(candidate.get("id", "")) for candidate in candidates]
    if not candidates or any(not item or any(char.isspace() for char in item) for item in ids):
        raise ValueError("Every Stage 3 candidate needs a whitespace-free id")
    if len(ids) != len(set(ids)):
        raise ValueError("Stage 3 candidate ids must be unique")
    for candidate in candidates:
        ratio = float(candidate.get("reasoning_fraction", -1.0))
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Candidate reasoning_fraction must be between 0 and 1")
        if str(candidate.get("loss_type")) not in ALLOWED_LOSSES:
            raise ValueError(f"Candidate loss_type must be one of {sorted(ALLOWED_LOSSES)}")
        if str(candidate.get("prompt_conditioning", "shared_flexible")) not in {
            "shared_flexible",
            "contract_conditioned",
        }:
            raise ValueError("Unknown candidate prompt_conditioning mode")
        if float(candidate.get("learning_rate", 0.0)) <= 0.0:
            raise ValueError("Candidate learning_rate must be positive")
        if int(candidate.get("max_steps", 0)) <= 0:
            raise ValueError("Candidate max_steps must be positive")


def _role(metrics: dict[str, Any]) -> str:
    return str(metrics.get("evaluation_input", {}).get("role", ""))


def _overall(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["overall"][key])


def _worst_group_delta(
    base: dict[str, Any], candidate: dict[str, Any]
) -> float:
    deltas: list[float] = []
    for group_name in ("by_family", "by_difficulty"):
        if set(base[group_name]) != set(candidate[group_name]):
            raise ValueError(f"Mismatched validation groups in {group_name}")
        for group in base[group_name]:
            deltas.append(
                float(candidate[group_name][group]["answer_exact_match"])
                - float(base[group_name][group]["answer_exact_match"])
            )
    return min(deltas)


def select_candidate(
    *,
    base_synthetic: dict[str, Any],
    base_external: dict[str, Any],
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Apply a deterministic validation-only selector to normalized run records."""

    if _role(base_synthetic) != "synthetic_validation":
        raise ValueError("Synthetic base selection input is not validation data")
    if _role(base_external) != "external_train_validation":
        raise ValueError("External base selection input is not train-derived validation data")

    candidate_ids = [str(candidate["id"]) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate selection records contain duplicate ids")

    max_synthetic_regression = float(policy["max_synthetic_regression"])
    max_external_regression = float(policy["max_external_regression"])
    max_group_regression = float(policy["max_group_regression"])
    min_format = float(policy["min_format_valid_rate"])
    min_parse = float(policy["min_parse_success_rate"])
    max_synth_limit = float(policy["max_synthetic_token_limit_rate"])
    max_external_limit = float(policy["max_external_token_limit_rate"])
    expected_seeds = sorted(int(seed) for seed in policy["expected_seeds"])

    base_synth_em = _overall(base_synthetic, "answer_exact_match")
    base_external_em = _overall(base_external, "answer_exact_match")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        seeds = candidate.get("seeds", [])
        observed_seeds = sorted(int(seed["seed"]) for seed in seeds)
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"Candidate {candidate['id']} seed set {observed_seeds} does not match "
                f"the preregistered set {expected_seeds}"
            )
        seed_records: list[dict[str, Any]] = []
        for seed in seeds:
            synthetic = seed["synthetic"]
            external = seed["external"]
            if _role(synthetic) != "synthetic_validation" or _role(
                external
            ) != "external_train_validation":
                raise ValueError("Candidate selection attempted to consume a non-validation role")
            synthetic_delta = _overall(synthetic, "answer_exact_match") - base_synth_em
            external_delta = _overall(external, "answer_exact_match") - base_external_em
            group_delta = _worst_group_delta(base_synthetic, synthetic)
            checks = {
                "synthetic_retention": synthetic_delta >= -max_synthetic_regression,
                "external_retention": external_delta >= -max_external_regression,
                "worst_group_retention": group_delta >= -max_group_regression,
                "format": _overall(synthetic, "format_valid_rate") >= min_format,
                "synthetic_parse": _overall(synthetic, "parse_success_rate") >= min_parse,
                "external_parse": _overall(external, "parse_success_rate") >= min_parse,
                "synthetic_token_limit": _overall(
                    synthetic, "hit_max_new_tokens_rate"
                )
                <= max_synth_limit,
                "external_token_limit": _overall(
                    external, "hit_max_new_tokens_rate"
                )
                <= max_external_limit,
            }
            seed_records.append(
                {
                    "seed": int(seed["seed"]),
                    "eligible": all(checks.values()),
                    "checks": checks,
                    "synthetic_exact": _overall(synthetic, "answer_exact_match"),
                    "synthetic_strict": _overall(synthetic, "strict_correct_rate"),
                    "synthetic_delta": synthetic_delta,
                    "external_exact": _overall(external, "answer_exact_match"),
                    "external_delta": external_delta,
                    "worst_group_delta": group_delta,
                }
            )
        eligible = all(record["eligible"] for record in seed_records)
        retention_values = [
            value
            for record in seed_records
            for value in (record["synthetic_delta"], record["external_delta"])
        ]
        records.append(
            {
                "id": str(candidate["id"]),
                "eligible": eligible,
                "seed_results": seed_records,
                "ranking": {
                    "worst_seed_retention": min(retention_values),
                    "mean_synthetic_strict": statistics.fmean(
                        record["synthetic_strict"] for record in seed_records
                    ),
                    "mean_external_exact": statistics.fmean(
                        record["external_exact"] for record in seed_records
                    ),
                },
            }
        )

    eligible_records = [record for record in records if record["eligible"]]
    eligible_records.sort(
        key=lambda record: (
            -record["ranking"]["worst_seed_retention"],
            -record["ranking"]["mean_synthetic_strict"],
            -record["ranking"]["mean_external_exact"],
            record["id"],
        )
    )
    selected = eligible_records[0]["id"] if eligible_records else None
    return {
        "selected_candidate": selected,
        "status": "selected" if selected else "no_eligible_candidate",
        "records": records,
        "policy": policy,
        "policy_sha256": canonical_hash(policy),
        "selection_input_roles": sorted(VALIDATION_ROLES),
    }
