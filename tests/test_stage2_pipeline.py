from pathlib import Path

import pytest

from reasoning_lab.stage2_pipeline import (
    _external_evaluation_config,
    _prepare_new_run_dir,
    _readiness_gate,
    mark_incomplete_run_aborted,
)


def _config() -> dict:
    return {
        "training": {"max_steps": 64, "max_length": 128},
        "continuation": {"max_steps": 16},
        "evaluation": {"max_hit_max_new_tokens_rate": 0.05},
        "external_evaluation": {
            "test_size": 8,
            "batch_size": 2,
            "max_prompt_tokens": 2048,
            "max_new_tokens": 128,
            "stop_strings": ["\nQuestion:"],
            "bootstrap_samples": 100,
            "bootstrap_seed": 7,
            "max_hit_max_new_tokens_rate": 0.10,
        },
    }


def _metrics(
    *,
    count: int = 8,
    hit_rate: float = 0.0,
    parse_rate: float = 1.0,
    exact_match: float = 1.0,
) -> dict:
    return {
        "overall": {
            "count": count,
            "hit_max_new_tokens_rate": hit_rate,
            "parse_success_rate": parse_rate,
            "answer_exact_match": exact_match,
        }
    }


def test_external_evaluation_settings_are_isolated() -> None:
    config = _config()
    external = _external_evaluation_config(config)
    assert external is not config
    assert external["evaluation"]["split"] == "test_eval"
    assert external["evaluation"]["max_prompt_tokens"] == 2048
    assert config["evaluation"] == {"max_hit_max_new_tokens_rate": 0.05}


def test_stage2_run_directory_must_be_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    assert _prepare_new_run_dir(empty) == empty.resolve()
    (empty / "artifact.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _prepare_new_run_dir(empty)


def test_readiness_gate_checks_fresh_branch_and_lineage() -> None:
    revision = "a" * 40
    metrics = {name: _metrics() for name in ("base", "sft", "continued_sft")}
    readiness = _readiness_gate(
        _config(),
        internal_metrics=metrics,
        external_metrics=metrics,
        sft_summary={"global_step": 64},
        continued_sft_summary={
            "global_step": 16,
            "initial_global_step": 0,
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "resumed_from_checkpoint": None,
            "parent_base_model_revision": revision,
        },
        model_revision=revision,
    )
    assert readiness["passed"]
    assert readiness["status"] == "READY FOR A DPO PILOT"


def test_readiness_gate_rejects_format_only_sft_regression() -> None:
    revision = "a" * 40
    internal = {
        "base": _metrics(exact_match=0.95),
        "sft": _metrics(exact_match=0.70),
        "continued_sft": _metrics(exact_match=0.68),
    }
    external = {name: _metrics() for name in ("base", "sft", "continued_sft")}
    readiness = _readiness_gate(
        _config(),
        internal_metrics=internal,
        external_metrics=external,
        sft_summary={"global_step": 64},
        continued_sft_summary={
            "global_step": 16,
            "initial_global_step": 0,
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "resumed_from_checkpoint": None,
            "parent_base_model_revision": revision,
        },
        model_revision=revision,
    )
    assert not readiness["passed"]
    assert readiness["status"] == "NOT READY FOR DPO"
    assert not readiness["checks"][
        "SFT preserves format-independent synthetic capability"
    ]["passed"]


def test_interrupted_run_can_be_closed_without_deleting_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "partial.txt").write_text("keep", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        '{"status":"running","stages":{"evaluation":{"status":"running"}}}',
        encoding="utf-8",
    )
    manifest = mark_incomplete_run_aborted(
        run_dir=run_dir,
        reason="controlled test interruption",
    )
    assert manifest["status"] == "aborted"
    assert manifest["stages"]["evaluation"]["status"] == "aborted"
    assert (run_dir / "partial.txt").read_text(encoding="utf-8") == "keep"
    assert (run_dir / "ABORTED").is_file()
