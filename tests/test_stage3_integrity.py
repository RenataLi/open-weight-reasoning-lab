from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoning_lab.stage3_pipeline import (
    _assert_confirmation_unexposed,
    _claim_confirmation_exposure,
    _confirmation_gate,
    _verify_search_evidence,
    _write_confirmation_report,
    _write_search_report,
)
from reasoning_lab.training.continued_sft import sha256_directory


REVISION = "a" * 40


def _file(role: str, *, count: int = 8, digest: str | None = None) -> dict:
    return {
        "role": role,
        "count": count,
        "sha256": digest or ("1" if "synthetic" in role else "2") * 64,
        "row_id_set_sha256": ("3" if "synthetic" in role else "4") * 64,
        "path": f"{role}.jsonl",
    }


def _metrics(
    role: str,
    locked_file: dict,
    *,
    exact: float = 0.95,
    format_valid: float = 1.0,
    parse: float = 1.0,
    token_limit: float = 0.0,
) -> dict:
    return {
        "model_revision": REVISION,
        "evaluation_input": {
            "role": role,
            "count": locked_file["count"],
            "sha256": locked_file["sha256"],
            "row_id_set_sha256": locked_file["row_id_set_sha256"],
        },
        "overall": {
            "answer_exact_match": exact,
            "format_valid_rate": format_valid,
            "strict_correct_rate": min(exact, format_valid),
            "parse_success_rate": parse,
            "hit_max_new_tokens_rate": token_limit,
        },
    }


def _config(tmp_path: Path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "paths": {"runs_dir": "runs"},
        "model": {"id": "Qwen/Qwen3-1.7B-Base"},
        "continuation": {"max_steps": 16},
        "stage3": {
            "search_seeds": [1, 2, 3],
            "confirmation_policy": {
                "max_synthetic_regression": 0.05,
                "max_external_regression": 0.03,
                "max_synthetic_ci_regression": 0.05,
                "max_external_ci_regression": 0.03,
                "max_continued_regression": 0.05,
                "min_format_valid_rate": 0.95,
                "min_parse_success_rate": 0.99,
                "max_synthetic_token_limit_rate": 0.05,
                "max_external_token_limit_rate": 0.10,
                "max_synthetic_seed_range": 0.03,
                "max_external_seed_range": 0.04,
            },
        },
    }


def _adapter(path: Path, content: bytes) -> str:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(content)
    return sha256_directory(path)


def _gate_fixture(tmp_path: Path, seeds: tuple[int, ...] = (1, 2, 3)):
    config = _config(tmp_path)
    synthetic_file = _file("synthetic_confirmation")
    external_file = _file("external_confirmation")
    lock = {
        "selection_locked_at": "2026-01-01T00:00:00+00:00",
        "test_metrics_used_for_selection": False,
        "model_revision": REVISION,
        "confirmation_files": {
            "synthetic_confirmation": synthetic_file,
            "external_confirmation": external_file,
        },
        "locked_continuation": {"sha256": "5" * 64},
        "selected_adapters": [],
    }
    base = {
        "synthetic": _metrics("synthetic_confirmation", synthetic_file),
        "external": _metrics("external_confirmation", external_file, exact=0.70),
    }
    results = []
    comparisons = {}
    for seed in seeds:
        parent = tmp_path / f"parent_{seed}"
        parent_hash = _adapter(parent, f"parent-{seed}".encode())
        continued = tmp_path / f"continued_{seed}"
        continued_hash = _adapter(continued, f"continued-{seed}".encode())
        locked_adapter = {"seed": seed, "path": str(parent), "sha256": parent_hash}
        lock["selected_adapters"].append(locked_adapter)
        sft_synth = _metrics("synthetic_confirmation", synthetic_file)
        sft_external = _metrics("external_confirmation", external_file, exact=0.70)
        cont_synth = _metrics("synthetic_confirmation", synthetic_file)
        cont_external = _metrics("external_confirmation", external_file, exact=0.70)
        results.append(
            {
                "seed": seed,
                "sft": {"synthetic": sft_synth, "external": sft_external},
                "continued_sft": {
                    "synthetic": cont_synth,
                    "external": cont_external,
                },
                "continued_training": {
                    "parent_adapter": str(parent),
                    "parent_adapter_sha256": parent_hash,
                    "parent_base_model_id": config["model"]["id"],
                    "parent_base_model_revision": REVISION,
                    "continuation_dataset_sha256": lock["locked_continuation"]["sha256"],
                    "fresh_optimizer": True,
                    "fresh_scheduler": True,
                    "initial_global_step": 0,
                    "resumed_from_checkpoint": None,
                    "global_step": 16,
                    "resolved_training_settings": {
                        "seed": seed + 10_000,
                        "data_seed": seed + 10_000,
                    },
                    "final_adapter": str(continued),
                    "final_adapter_sha256": continued_hash,
                },
            }
        )
        comparison = {
            "paired_bootstrap_95_ci": [-0.01, 0.01],
        }
        comparisons[f"seed_{seed}"] = {
            "base_to_sft_synthetic": comparison,
            "base_to_sft_external": comparison,
        }
    return config, lock, base, results, comparisons


def test_confirmation_gate_rejects_missing_or_duplicate_seed(tmp_path: Path) -> None:
    config, lock, base, results, comparisons = _gate_fixture(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        _confirmation_gate(
            config,
            lock=lock,
            marker_time="2026-01-01T00:01:00+00:00",
            base=base,
            seeds=results[:-1],
            comparisons=comparisons,
        )
    duplicated = [results[0], results[0], results[2]]
    with pytest.raises(ValueError, match="duplicate seeds"):
        _confirmation_gate(
            config,
            lock=lock,
            marker_time="2026-01-01T00:01:00+00:00",
            base=base,
            seeds=duplicated,
            comparisons=comparisons,
        )


def test_confirmation_gate_rejects_tampered_metric_provenance(tmp_path: Path) -> None:
    config, lock, base, results, comparisons = _gate_fixture(tmp_path)
    results[0]["sft"]["synthetic"]["evaluation_input"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="provenance mismatch"):
        _confirmation_gate(
            config,
            lock=lock,
            marker_time="2026-01-01T00:01:00+00:00",
            base=base,
            seeds=results,
            comparisons=comparisons,
        )


def test_confirmation_gate_checks_continued_quality_and_lineage(tmp_path: Path) -> None:
    config, lock, base, results, comparisons = _gate_fixture(tmp_path)
    readiness = _confirmation_gate(
        config,
        lock=lock,
        marker_time="2026-01-01T00:01:00+00:00",
        base=base,
        seeds=results,
        comparisons=comparisons,
    )
    assert readiness["passed"]

    results[0]["continued_sft"]["synthetic"]["overall"]["format_valid_rate"] = 0.5
    results[0]["continued_training"]["parent_adapter_sha256"] = "0" * 64
    readiness = _confirmation_gate(
        config,
        lock=lock,
        marker_time="2026-01-01T00:01:00+00:00",
        base=base,
        seeds=results,
        comparisons=comparisons,
    )
    assert not readiness["passed"]
    detail = readiness["checks"]["Seed 1 passes capability and format thresholds"]["detail"]
    assert not detail["continued_format"]
    assert not detail["continued_lineage"]


def test_confirmation_exposure_is_single_use_and_recognizes_legacy_runs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    files = {
        "synthetic_confirmation": _file("synthetic_confirmation"),
        "external_confirmation": _file("external_confirmation"),
    }
    lock = {"confirmation_files": files}
    claim = _claim_confirmation_exposure(
        config,
        lock=lock,
        lock_hash="9" * 64,
        search_run=tmp_path / "search",
        confirmation_run=tmp_path / "confirm",
    )
    assert Path(claim["registry_path"]).is_file()
    with pytest.raises(FileExistsError, match="already claimed"):
        _claim_confirmation_exposure(
            config,
            lock=lock,
            lock_hash="9" * 64,
            search_run=tmp_path / "search",
            confirmation_run=tmp_path / "confirm2",
        )

    other_config = _config(tmp_path / "legacy")
    old = Path(other_config["_project_root"]) / "runs" / "old-confirm"
    old.mkdir(parents=True)
    (old / "run_manifest.json").write_text(
        json.dumps(
            {
                "phase": "confirmation",
                "test_evaluation_started_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (old / "selection_lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="earlier run"):
        _assert_confirmation_unexposed(other_config, files)


def test_search_evidence_tamper_is_detected(tmp_path: Path) -> None:
    search = tmp_path / "search"
    search.mkdir()
    evidence = search / "metrics.json"
    evidence.write_text("original", encoding="utf-8")
    import hashlib

    lock = {
        "search_evidence_sha256": {
            "metrics.json": hashlib.sha256(evidence.read_bytes()).hexdigest()
        }
    }
    _verify_search_evidence(search, lock)
    evidence.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        _verify_search_evidence(search, lock)


def test_stage3_reports_are_data_driven_and_distinguish_format(tmp_path: Path) -> None:
    search_report = tmp_path / "search.md"
    search_config = {
        "stage3": {
            "search_seeds": [1, 2, 3],
            "candidates": [
                {"loss_type": "nll", "max_steps": 32},
                {"loss_type": "nll", "max_steps": 32},
            ],
        }
    }
    _write_search_report(
        search_report,
        config=search_config,
        model_revision=REVISION,
        selection={
            "status": "selected",
            "selected_candidate": "x",
            "records": [
                {
                    "id": "x",
                    "eligible": True,
                    "ranking": {
                        "worst_seed_retention": 0.0,
                        "mean_synthetic_strict": 0.9,
                        "mean_external_exact": 0.7,
                    },
                }
            ],
        },
        base_synthetic={"overall": {"answer_exact_match": 0.9}},
        base_external={"overall": {"answer_exact_match": 0.7}},
        frozen_manifest={
            "synthetic_contract": "reasoning_then_answer",
            "files": {
                "synthetic_validation": {"count": 5},
                "synthetic_confirmation": {"count": 7},
                "external_train_validation": {"count": 6},
                "external_confirmation": {"count": 9},
            },
        },
    )
    text = search_report.read_text(encoding="utf-8")
    assert "includes DFT" not in text
    assert "All candidates in this round use NLL" in text

    confirmation_report = tmp_path / "confirmation.md"
    synth_file = _file("synthetic_confirmation", count=7)
    external_file = _file("external_confirmation", count=9)
    metric_synth = _metrics(
        "synthetic_confirmation", synth_file, exact=0.8, format_valid=1.0
    )
    metric_external = _metrics("external_confirmation", external_file, exact=0.7)
    _write_confirmation_report(
        confirmation_report,
        lock={
            "selected_candidate": "x",
            "model_revision": REVISION,
            "confirmation_files": {
                "synthetic_confirmation": synth_file,
                "external_confirmation": external_file,
            },
        },
        base={"synthetic": metric_synth, "external": metric_external},
        seeds=[
            {
                "seed": 1,
                "sft": {"synthetic": metric_synth, "external": metric_external},
                "continued_sft": {
                    "synthetic": metric_synth,
                    "external": metric_external,
                },
            }
        ],
        readiness={
            "status": "READY",
            "checks": {},
            "summary": {
                "sft_synthetic_exact_mean": 0.8,
                "sft_synthetic_delta_mean": 0.0,
                "sft_external_exact_mean": 0.7,
                "sft_external_delta_mean": 0.0,
            },
        },
    )
    text = confirmation_report.read_text(encoding="utf-8")
    assert "7 new synthetic examples and 9 GSM8K" in text
    assert "Synthetic format valid" in text
    assert "Synthetic strict-correct" in text
    assert "| SFT | 1 | 80.0% | 100.0% | 80.0% | 70.0% |" in text
