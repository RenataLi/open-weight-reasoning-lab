from __future__ import annotations

import copy
from pathlib import Path

import pytest

from reasoning_lab.config import load_config
from reasoning_lab.stage4_selection import validate_stage4_spec


def test_stage4_project_config_is_valid() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "stage4_dpo_qwen3_1_7b.yaml")
    assert config["stage4"]["parent_seed"] == 101
    assert {candidate["loss_type"] for candidate in config["stage4"]["candidates"]} == {
        "sigmoid"
    }
    anchored = next(
        candidate
        for candidate in config["stage4"]["candidates"]
        if candidate.get("auxiliary_loss_type") == "sft"
    )
    assert anchored["loss_weights"] == [1.0, 0.25]
    assert config["stage4"]["dpo_training"]["disable_dropout"] is True
    assert (
        config["stage4"]["svamp_validation_size"]
        + config["stage4"]["svamp_development_quarantine_size"]
        == 999
    )
    assert "svamp_confirmation_size" not in config["stage4"]
    assert "svamp_retired_size" not in config["stage4"]
    assert not {
        key
        for key in config["stage4"]["confirmation_policy"]
        if "external" in key or key == "objective_benefit_other_ci_floor"
    }


def test_stage4_spec_forbids_svamp_confirmation_role() -> None:
    project_root = Path(__file__).resolve().parents[1]
    stage4 = load_config(project_root / "configs" / "stage4_dpo_qwen3_1_7b.yaml")[
        "stage4"
    ]
    legacy = copy.deepcopy(stage4)
    legacy["svamp_confirmation_size"] = legacy.pop(
        "svamp_development_quarantine_size"
    )

    with pytest.raises(ValueError, match="forbids SVAMP confirmation"):
        validate_stage4_spec(legacy)


def test_stage4_confirmation_policy_forbids_external_endpoints() -> None:
    project_root = Path(__file__).resolve().parents[1]
    stage4 = copy.deepcopy(
        load_config(project_root / "configs" / "stage4_dpo_qwen3_1_7b.yaml")[
            "stage4"
        ]
    )
    stage4["confirmation_policy"]["max_external_regression"] = 0.03

    with pytest.raises(ValueError, match="synthetic-only"):
        validate_stage4_spec(stage4)
