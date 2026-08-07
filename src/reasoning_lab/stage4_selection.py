from __future__ import annotations

import hashlib
import json
from typing import Any


VALIDATION_ROLES = {
    "stage4_synthetic_validation",
    "stage4_svamp_validation",
    "stage4_preference_validation",
}
ALLOWED_DPO_LOSSES = {"sigmoid", "sigmoid_norm"}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_stage4_spec(stage4: dict[str, Any]) -> None:
    forbidden_sizes = {"svamp_retired_size", "svamp_confirmation_size"} & set(stage4)
    if forbidden_sizes:
        raise ValueError(
            "Stage 4 forbids SVAMP confirmation/retired roles because all non-validation "
            f"rows are development-exposed: {sorted(forbidden_sizes)}"
        )
    required_sizes = (
        "preference_train_size",
        "preference_validation_size",
        "preference_confirmation_size",
        "synthetic_validation_size",
        "synthetic_confirmation_size",
        "svamp_validation_size",
        "svamp_development_quarantine_size",
    )
    for name in required_sizes:
        if int(stage4.get(name, 0)) <= 0:
            raise ValueError(f"stage4.{name} must be positive")
    if (
        int(stage4["svamp_validation_size"])
        + int(stage4["svamp_development_quarantine_size"])
        != 999
    ):
        raise ValueError(
            "Stage 4 SVAMP validation plus development quarantine must account for all 999 rows"
        )
    parent_hash = str(stage4.get("parent_adapter_sha256", ""))
    if len(parent_hash) != 64:
        raise ValueError("stage4.parent_adapter_sha256 must be a SHA-256 digest")
    candidates = list(stage4.get("candidates", []))
    if not candidates:
        raise ValueError("Stage 4 requires at least one DPO candidate")
    ids = [str(candidate.get("id", "")) for candidate in candidates]
    if any(not item or any(char.isspace() for char in item) for item in ids):
        raise ValueError("Every Stage 4 candidate needs a whitespace-free id")
    if len(ids) != len(set(ids)):
        raise ValueError("Stage 4 candidate ids must be unique")
    for candidate in candidates:
        if str(candidate.get("loss_type")) not in ALLOWED_DPO_LOSSES:
            raise ValueError(
                f"Stage 4 loss_type must be one of {sorted(ALLOWED_DPO_LOSSES)}"
            )
        if float(candidate.get("beta", 0.0)) <= 0.0:
            raise ValueError("Stage 4 candidate beta must be positive")
        if float(candidate.get("learning_rate", 0.0)) <= 0.0:
            raise ValueError("Stage 4 candidate learning_rate must be positive")
        if int(candidate.get("max_steps", 0)) <= 0:
            raise ValueError("Stage 4 candidate max_steps must be positive")
        auxiliary = candidate.get("auxiliary_loss_type")
        if auxiliary is not None and str(auxiliary) != "sft":
            raise ValueError("Stage 4 only supports 'sft' as an auxiliary loss")
        weights = candidate.get("loss_weights")
        expected_weight_count = 2 if auxiliary is not None else 1
        if weights is not None and (
            len(weights) != expected_weight_count
            or any(float(value) <= 0.0 for value in weights)
        ):
            raise ValueError(
                "Stage 4 loss_weights must be positive and match configured losses"
            )
    controls = list(stage4.get("sft_controls", []))
    if not controls:
        raise ValueError("Stage 4 requires at least one matched extra-SFT control")
    control_ids = [str(control.get("id", "")) for control in controls]
    if any(not item for item in control_ids) or len(control_ids) != len(set(control_ids)):
        raise ValueError("Stage 4 control ids must be present and unique")
    if not set(stage4.get("selection_policy", {})):
        raise ValueError("Missing stage4.selection_policy")
    if not set(stage4.get("confirmation_policy", {})):
        raise ValueError("Missing stage4.confirmation_policy")
    if str(stage4.get("primary_benefit_endpoint")) != "synthetic":
        raise ValueError("stage4.primary_benefit_endpoint must be 'synthetic'")
    selection_policy_keys = {
        "max_synthetic_regression",
        "max_external_regression",
        "max_family_regression",
        "min_preference_accuracy",
        "min_synthetic_format_valid_rate",
        "min_external_format_valid_rate",
        "max_synthetic_format_regression",
        "max_external_format_regression",
        "max_synthetic_strict_regression",
        "max_external_strict_regression",
        "min_parse_success_rate",
        "max_synthetic_token_limit_rate",
        "max_external_token_limit_rate",
        "min_length_ratio",
        "max_length_ratio",
    }
    confirmation_policy_keys = {
        "max_synthetic_regression",
        "max_synthetic_ci_regression",
        "max_family_regression",
        "min_preference_accuracy",
        "min_preference_accuracy_advantage_vs_control",
        "min_synthetic_format_valid_rate",
        "max_synthetic_format_regression",
        "max_synthetic_strict_regression",
        "min_parse_success_rate",
        "max_synthetic_token_limit_rate",
        "min_length_ratio",
        "max_length_ratio",
        "objective_benefit_delta",
    }
    selection_policy = stage4["selection_policy"]
    confirmation_policy = stage4["confirmation_policy"]
    missing_selection = selection_policy_keys - set(selection_policy)
    missing_confirmation = confirmation_policy_keys - set(confirmation_policy)
    if missing_selection:
        raise ValueError(f"Missing Stage 4 selection policy keys: {sorted(missing_selection)}")
    if missing_confirmation:
        raise ValueError(
            f"Missing Stage 4 confirmation policy keys: {sorted(missing_confirmation)}"
        )
    forbidden_confirmation = {
        "max_external_regression",
        "max_external_ci_regression",
        "min_external_format_valid_rate",
        "max_external_format_regression",
        "max_external_strict_regression",
        "max_external_token_limit_rate",
        "objective_benefit_other_ci_floor",
    } & set(confirmation_policy)
    if forbidden_confirmation:
        raise ValueError(
            "Stage 4 confirmation policy must be synthetic-only; remove: "
            f"{sorted(forbidden_confirmation)}"
        )
    for name, policy in (
        ("selection", selection_policy),
        ("confirmation", confirmation_policy),
    ):
        if not 0.0 < float(policy["min_length_ratio"]) <= 1.0:
            raise ValueError(f"Stage 4 {name} min_length_ratio must be in (0, 1]")
        if float(policy["max_length_ratio"]) < 1.0:
            raise ValueError(f"Stage 4 {name} max_length_ratio must be at least 1")


def _role(metrics: dict[str, Any]) -> str:
    return str(metrics.get("evaluation_input", {}).get("role", ""))


def _overall(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["overall"][key])


def _preference_value(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["objective"]["overall"][key])


def _worst_family_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> float:
    if set(baseline["by_family"]) != set(candidate["by_family"]):
        raise ValueError("Mismatched family groups in Stage 4 validation metrics")
    return min(
        float(candidate["by_family"][family]["answer_exact_match"])
        - float(baseline["by_family"][family]["answer_exact_match"])
        for family in baseline["by_family"]
    )


def _length_ratio(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    denominator = _overall(baseline, "generated_tokens_mean")
    numerator = _overall(candidate, "generated_tokens_mean")
    if denominator <= 0.0:
        raise ValueError("Baseline generated-token mean must be positive")
    return numerator / denominator


def select_control(
    controls: list[dict[str, Any]], *, expected_ids: list[str]
) -> dict[str, Any]:
    observed = sorted(str(control["id"]) for control in controls)
    if observed != sorted(expected_ids):
        raise ValueError("Observed extra-SFT controls differ from preregistration")
    for control in controls:
        if _role(control["synthetic"]) != "stage4_synthetic_validation":
            raise ValueError("Control selector consumed non-validation synthetic data")
        if _role(control["external"]) != "stage4_svamp_validation":
            raise ValueError("Control selector consumed non-validation SVAMP data")
    ranked = sorted(
        controls,
        key=lambda control: (
            -min(
                _overall(control["synthetic"], "answer_exact_match"),
                _overall(control["external"], "answer_exact_match"),
            ),
            -(
                _overall(control["synthetic"], "answer_exact_match")
                + _overall(control["external"], "answer_exact_match")
            ),
            str(control["id"]),
        ),
    )
    return {
        "selected_control": str(ranked[0]["id"]),
        "ranking": [str(control["id"]) for control in ranked],
    }


def select_dpo_candidate(
    *,
    base_synthetic: dict[str, Any],
    base_external: dict[str, Any],
    candidates: list[dict[str, Any]],
    expected_candidate_ids: list[str],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Apply a deterministic selector that accepts development roles only."""

    if _role(base_synthetic) != "stage4_synthetic_validation":
        raise ValueError("Stage 4 base synthetic input is not validation data")
    if _role(base_external) != "stage4_svamp_validation":
        raise ValueError("Stage 4 base external input is not validation data")
    observed = sorted(str(candidate["id"]) for candidate in candidates)
    if observed != sorted(expected_candidate_ids):
        raise ValueError("Observed DPO candidates differ from preregistration")

    base_synth_exact = _overall(base_synthetic, "answer_exact_match")
    base_external_exact = _overall(base_external, "answer_exact_match")
    base_synth_format = _overall(base_synthetic, "format_valid_rate")
    base_external_format = _overall(base_external, "format_valid_rate")
    base_synth_strict = _overall(base_synthetic, "strict_correct_rate")
    base_external_strict = _overall(base_external, "strict_correct_rate")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        synthetic = candidate["synthetic"]
        external = candidate["external"]
        preference = candidate["preference"]
        if _role(synthetic) != "stage4_synthetic_validation":
            raise ValueError("DPO selector consumed non-validation synthetic data")
        if _role(external) != "stage4_svamp_validation":
            raise ValueError("DPO selector consumed non-validation SVAMP data")
        if _role(preference) != "stage4_preference_validation":
            raise ValueError("DPO selector consumed non-validation preference data")

        synth_delta = _overall(synthetic, "answer_exact_match") - base_synth_exact
        external_delta = _overall(external, "answer_exact_match") - base_external_exact
        worst_family = min(
            _worst_family_delta(base_synthetic, synthetic),
            _worst_family_delta(base_external, external),
        )
        synth_length_ratio = _length_ratio(base_synthetic, synthetic)
        external_length_ratio = _length_ratio(base_external, external)
        preference_accuracy = _preference_value(preference, "reward_accuracy")
        preference_margin = _preference_value(preference, "reward_margin_mean")
        checks = {
            "synthetic_retention": synth_delta
            >= -float(policy["max_synthetic_regression"]),
            "external_retention": external_delta
            >= -float(policy["max_external_regression"]),
            "worst_family_retention": worst_family
            >= -float(policy["max_family_regression"]),
            "preference_accuracy": preference_accuracy
            >= float(policy["min_preference_accuracy"]),
            "preference_margin": preference_margin > 0.0,
            "synthetic_format": _overall(synthetic, "format_valid_rate")
            >= max(
                float(policy["min_synthetic_format_valid_rate"]),
                base_synth_format
                - float(policy["max_synthetic_format_regression"]),
            ),
            "external_format": _overall(external, "format_valid_rate")
            >= max(
                float(policy["min_external_format_valid_rate"]),
                base_external_format
                - float(policy["max_external_format_regression"]),
            ),
            "synthetic_strict_retention": _overall(
                synthetic, "strict_correct_rate"
            )
            >= base_synth_strict - float(policy["max_synthetic_strict_regression"]),
            "external_strict_retention": _overall(external, "strict_correct_rate")
            >= base_external_strict
            - float(policy["max_external_strict_regression"]),
            "synthetic_parse": _overall(synthetic, "parse_success_rate")
            >= float(policy["min_parse_success_rate"]),
            "external_parse": _overall(external, "parse_success_rate")
            >= float(policy["min_parse_success_rate"]),
            "synthetic_token_limit": _overall(
                synthetic, "hit_max_new_tokens_rate"
            )
            <= float(policy["max_synthetic_token_limit_rate"]),
            "external_token_limit": _overall(external, "hit_max_new_tokens_rate")
            <= float(policy["max_external_token_limit_rate"]),
            "synthetic_length": float(policy["min_length_ratio"])
            <= synth_length_ratio
            <= float(policy["max_length_ratio"]),
            "external_length": float(policy["min_length_ratio"])
            <= external_length_ratio
            <= float(policy["max_length_ratio"]),
        }
        records.append(
            {
                "id": str(candidate["id"]),
                "eligible": all(checks.values()),
                "checks": checks,
                "diagnostics": {
                    "synthetic_delta": synth_delta,
                    "external_delta": external_delta,
                    "worst_family_delta": worst_family,
                    "preference_accuracy": preference_accuracy,
                    "preference_margin": preference_margin,
                    "synthetic_format_delta": _overall(
                        synthetic, "format_valid_rate"
                    )
                    - base_synth_format,
                    "external_format_delta": _overall(
                        external, "format_valid_rate"
                    )
                    - base_external_format,
                    "synthetic_strict_delta": _overall(
                        synthetic, "strict_correct_rate"
                    )
                    - base_synth_strict,
                    "external_strict_delta": _overall(
                        external, "strict_correct_rate"
                    )
                    - base_external_strict,
                    "synthetic_length_ratio": synth_length_ratio,
                    "external_length_ratio": external_length_ratio,
                },
                "ranking": {
                    "worst_task_retention": min(synth_delta, external_delta, worst_family),
                    "synthetic_exact": _overall(synthetic, "answer_exact_match"),
                    "external_exact": _overall(external, "answer_exact_match"),
                    "preference_accuracy": preference_accuracy,
                    "length_drift": max(
                        abs(synth_length_ratio - 1.0),
                        abs(external_length_ratio - 1.0),
                    ),
                },
            }
        )

    eligible = [record for record in records if record["eligible"]]
    eligible.sort(
        key=lambda record: (
            -record["ranking"]["worst_task_retention"],
            -record["ranking"]["synthetic_exact"],
            -record["ranking"]["external_exact"],
            -record["ranking"]["preference_accuracy"],
            record["ranking"]["length_drift"],
            record["id"],
        )
    )
    selected = eligible[0]["id"] if eligible else None
    return {
        "status": "selected" if selected else "no_eligible_candidate",
        "selected_candidate": selected,
        "records": records,
        "selection_input_roles": sorted(VALIDATION_ROLES),
        "policy": policy,
        "policy_sha256": canonical_hash(policy),
    }
