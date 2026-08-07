from __future__ import annotations

import copy
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import yaml
from huggingface_hub import HfApi

from reasoning_lab.config import public_config, resolve_path
from reasoning_lab.data.arithmetic import generate_dataset
from reasoning_lab.data.gsm8k import prepare_gsm8k_snapshot
from reasoning_lab.evaluation.generation import evaluate_model
from reasoning_lab.evaluation.metrics import compare_predictions
from reasoning_lab.io_utils import read_json, read_jsonl, sha256_file, write_json
from reasoning_lab.reporting.stage2 import write_stage2_report
from reasoning_lab.reproducibility import environment_snapshot, run_gpu_preflight
from reasoning_lab.training.continued_sft import train_continued_sft
from reasoning_lab.training.sft import train_sft


EXACT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _new_run_dir(config: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return resolve_path(config, "runs_dir") / f"{timestamp}_{_safe_name(config['project']['run_name'])}"


def _prepare_new_run_dir(path: str | Path) -> Path:
    destination = Path(path).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Stage 2 only supports a new empty run directory; refusing to overwrite {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _resolve_model_revision(config: dict[str, Any]) -> str:
    requested = str(config["model"]["revision"])
    if not EXACT_REVISION_PATTERN.fullmatch(requested):
        raise ValueError("Stage 2 model.revision must be a full 40-character commit SHA")
    info = HfApi().model_info(repo_id=config["model"]["id"], revision=requested)
    resolved = str(info.sha or "")
    if resolved.lower() != requested.lower():
        raise RuntimeError(
            f"Resolved model revision {resolved!r} does not match pinned revision {requested!r}"
        )
    return resolved.lower()


def _resolve_dataset_revision(config: dict[str, Any]) -> str:
    external = config["external_evaluation"]
    requested = str(external["revision"])
    if not EXACT_REVISION_PATTERN.fullmatch(requested):
        raise ValueError(
            "Stage 2 external_evaluation.revision must be a full 40-character commit SHA"
        )
    info = HfApi().dataset_info(repo_id=external["dataset_id"], revision=requested)
    resolved = str(info.sha or "")
    if resolved.lower() != requested.lower():
        raise RuntimeError(
            f"Resolved dataset revision {resolved!r} does not match pinned revision {requested!r}"
        )
    return resolved.lower()


def _external_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    external = config["external_evaluation"]
    cloned = copy.deepcopy(config)
    cloned["evaluation"] = {
        "split": "test_eval",
        "max_examples": int(external["test_size"]),
        "batch_size": int(external["batch_size"]),
        "max_prompt_tokens": int(external["max_prompt_tokens"]),
        "max_new_tokens": int(external["max_new_tokens"]),
        "load_in_4bit": bool(external.get("load_in_4bit", False)),
        "stop_strings": list(external.get("stop_strings", [])),
        "bootstrap_samples": int(external["bootstrap_samples"]),
        "bootstrap_seed": int(external["bootstrap_seed"]),
        "max_hit_max_new_tokens_rate": float(
            external["max_hit_max_new_tokens_rate"]
        ),
    }
    return cloned


def _compare(
    left_path: Path,
    right_path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return compare_predictions(
        read_jsonl(left_path),
        read_jsonl(right_path),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _readiness_gate(
    config: dict[str, Any],
    *,
    internal_metrics: dict[str, dict[str, Any]],
    external_metrics: dict[str, dict[str, Any]],
    sft_summary: dict[str, Any],
    continued_sft_summary: dict[str, Any],
    model_revision: str,
) -> dict[str, Any]:
    internal_limit = float(config["evaluation"]["max_hit_max_new_tokens_rate"])
    external_limit = float(
        config["external_evaluation"]["max_hit_max_new_tokens_rate"]
    )
    internal_worst = max(
        float(metrics["overall"]["hit_max_new_tokens_rate"])
        for metrics in internal_metrics.values()
    )
    external_worst = max(
        float(metrics["overall"]["hit_max_new_tokens_rate"])
        for metrics in external_metrics.values()
    )
    external_parse_worst = min(
        float(metrics["overall"]["parse_success_rate"])
        for metrics in external_metrics.values()
    )
    expected_external_count = int(config["external_evaluation"]["test_size"])
    observed_counts = {
        int(metrics["overall"]["count"]) for metrics in external_metrics.values()
    }
    base_internal_em = float(internal_metrics["base"]["overall"]["answer_exact_match"])
    sft_internal_em = float(internal_metrics["sft"]["overall"]["answer_exact_match"])
    continued_internal_em = float(
        internal_metrics["continued_sft"]["overall"]["answer_exact_match"]
    )
    checks = {
        "Synthetic generation is not length-confounded": {
            "passed": internal_worst <= internal_limit,
            "detail": f"worst token-limit rate {internal_worst:.1%}, threshold {internal_limit:.1%}",
        },
        "External generation is not length-confounded": {
            "passed": external_worst <= external_limit,
            "detail": f"worst token-limit rate {external_worst:.1%}, threshold {external_limit:.1%}",
        },
        "External parser is reliable": {
            "passed": external_parse_worst >= 0.90,
            "detail": f"lowest parse-success rate {external_parse_worst:.1%}, threshold 90.0%",
        },
        "Frozen external subset is complete": {
            "passed": observed_counts == {expected_external_count},
            "detail": f"observed counts {sorted(observed_counts)}, expected {expected_external_count}",
        },
        "Initial SFT completed as configured": {
            "passed": int(sft_summary["global_step"]) == int(config["training"]["max_steps"]),
            "detail": f"{sft_summary['global_step']} / {config['training']['max_steps']} optimizer steps",
        },
        "SFT preserves format-independent synthetic capability": {
            "passed": sft_internal_em - base_internal_em >= -0.05,
            "detail": (
                f"Base {base_internal_em:.1%}, SFT {sft_internal_em:.1%}, "
                f"delta {sft_internal_em - base_internal_em:+.1%}; allowed regression 5.0%"
            ),
        },
        "Continued-SFT does not materially regress its SFT parent": {
            "passed": continued_internal_em - sft_internal_em >= -0.05,
            "detail": (
                f"SFT {sft_internal_em:.1%}, continued-SFT {continued_internal_em:.1%}, "
                f"delta {continued_internal_em - sft_internal_em:+.1%}; "
                "allowed regression 5.0%"
            ),
        },
        "Continued-SFT is a fresh matched branch": {
            "passed": (
                int(continued_sft_summary["global_step"])
                == int(config["continuation"]["max_steps"])
                and int(continued_sft_summary.get("initial_global_step", -1)) == 0
                and bool(continued_sft_summary.get("fresh_optimizer"))
                and bool(continued_sft_summary.get("fresh_scheduler"))
                and continued_sft_summary.get("resumed_from_checkpoint") is None
            ),
            "detail": "parent adapter loaded with fresh optimizer, scheduler, and global step",
        },
        "Parent adapter lineage is exact": {
            "passed": str(
                continued_sft_summary.get("parent_base_model_revision", "")
            ).lower()
            == model_revision.lower(),
            "detail": f"recorded base revision {continued_sft_summary.get('parent_base_model_revision')}",
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "status": "READY FOR A DPO PILOT" if passed else "NOT READY FOR DPO",
        "passed": passed,
        "checks": checks,
    }


def _primary_artifact_hashes(destination: Path) -> dict[str, str]:
    roots = [
        destination / "environment.json",
        destination / "resolved_config.yaml",
        destination / "data_manifest.json",
        destination / "external_data_manifest.json",
        destination / "comparisons.json",
        destination / "readiness.json",
        destination / "report.md",
        destination / "sft" / "training_summary.json",
        destination / "continued_sft" / "training_summary.json",
    ]
    for model_name in ("base", "sft", "continued_sft"):
        for benchmark_name in ("synthetic", "gsm8k"):
            roots.extend(
                [
                    destination / "evaluation" / model_name / benchmark_name / "metrics.json",
                    destination
                    / "evaluation"
                    / model_name
                    / benchmark_name
                    / "predictions.jsonl",
                ]
            )
    missing = [str(path) for path in roots if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing primary Stage 2 artifacts: {missing}")
    return {
        path.relative_to(destination).as_posix(): sha256_file(path) for path in roots
    }


def run_stage2_pipeline(
    config: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
) -> Path:
    if "external_evaluation" not in config or "continuation" not in config:
        raise ValueError("Stage 2 requires external_evaluation and continuation sections")
    destination = _prepare_new_run_dir(run_dir or _new_run_dir(config))
    cache_dir = resolve_path(config, "cache_dir")
    dataset_cache_dir = resolve_path(config, "dataset_cache_dir")
    data_dir = resolve_path(config, "data_dir")
    external_data_dir = resolve_path(config, "external_data_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": _utc_now(),
        "run_dir": str(destination),
        "config_hash": config["_config_hash"],
        "lifecycle": "new",
        "stages": {},
    }
    manifest_path = destination / "run_manifest.json"
    write_json(manifest_path, manifest)
    with (destination / "resolved_config.yaml").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        yaml.safe_dump(public_config(config), handle, sort_keys=True, allow_unicode=True)

    def stage(name: str, function: Callable[[], Any]) -> Any:
        started = perf_counter()
        manifest["stages"][name] = {"status": "running", "started_at": _utc_now()}
        write_json(manifest_path, manifest)
        try:
            result = function()
        except Exception:
            manifest["stages"][name].update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "duration_seconds": perf_counter() - started,
                }
            )
            write_json(manifest_path, manifest)
            raise
        manifest["stages"][name].update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "duration_seconds": perf_counter() - started,
            }
        )
        write_json(manifest_path, manifest)
        return result

    try:
        environment = stage("environment", environment_snapshot)
        write_json(destination / "environment.json", environment)
        manifest["preflight"] = stage("preflight", run_gpu_preflight)

        model_revision = stage("resolve_model_revision", lambda: _resolve_model_revision(config))
        dataset_revision = stage(
            "resolve_dataset_revision", lambda: _resolve_dataset_revision(config)
        )
        manifest["model"] = {
            "id": config["model"]["id"],
            "requested_revision": config["model"]["revision"],
            "resolved_revision": model_revision,
        }
        manifest["external_dataset"] = {
            "id": config["external_evaluation"]["dataset_id"],
            "requested_revision": config["external_evaluation"]["revision"],
            "resolved_revision": dataset_revision,
            "training_use": "forbidden",
        }

        data_manifest = stage("generate_synthetic_data", lambda: generate_dataset(config, data_dir))
        write_json(destination / "data_manifest.json", data_manifest)
        external_manifest = stage(
            "freeze_gsm8k_snapshot",
            lambda: prepare_gsm8k_snapshot(
                config,
                resolved_revision=dataset_revision,
                cache_dir=dataset_cache_dir,
                output_dir=external_data_dir,
            ),
        )
        write_json(destination / "external_data_manifest.json", external_manifest)

        internal_path = data_dir / f"{config['evaluation']['split']}.jsonl"
        external_path = external_data_dir / "test_eval.jsonl"
        external_config = _external_evaluation_config(config)
        evaluation_root = destination / "evaluation"

        internal_metrics: dict[str, dict[str, Any]] = {}
        external_metrics: dict[str, dict[str, Any]] = {}

        internal_metrics["base"] = stage(
            "evaluate_base_synthetic",
            lambda: evaluate_model(
                config,
                revision=model_revision,
                cache_dir=cache_dir,
                data_path=internal_path,
                output_dir=evaluation_root / "base" / "synthetic",
            ),
        )
        external_metrics["base"] = stage(
            "evaluate_base_gsm8k",
            lambda: evaluate_model(
                external_config,
                revision=model_revision,
                cache_dir=cache_dir,
                data_path=external_path,
                output_dir=evaluation_root / "base" / "gsm8k",
            ),
        )

        sft_root = destination / "sft"
        sft_summary = stage(
            "train_sft",
            lambda: train_sft(
                config,
                revision=model_revision,
                cache_dir=cache_dir,
                train_path=data_dir / "train.jsonl",
                output_dir=sft_root,
                resume=False,
            ),
        )
        sft_adapter = sft_root / "final_adapter"
        internal_metrics["sft"] = stage(
            "evaluate_sft_synthetic",
            lambda: evaluate_model(
                config,
                revision=model_revision,
                cache_dir=cache_dir,
                data_path=internal_path,
                output_dir=evaluation_root / "sft" / "synthetic",
                adapter_path=sft_adapter,
            ),
        )
        external_metrics["sft"] = stage(
            "evaluate_sft_gsm8k",
            lambda: evaluate_model(
                external_config,
                revision=model_revision,
                cache_dir=cache_dir,
                data_path=external_path,
                output_dir=evaluation_root / "sft" / "gsm8k",
                adapter_path=sft_adapter,
            ),
        )

        continued_root = destination / "continued_sft"
        continued_summary = stage(
            "train_continued_sft",
            lambda: train_continued_sft(
                config,
                revision=model_revision,
                cache_dir=cache_dir,
                continuation_path=data_dir / "continuation.jsonl",
                parent_adapter=sft_adapter,
                output_dir=continued_root,
            ),
        )
        continued_adapter = continued_root / "final_adapter"
        internal_metrics["continued_sft"] = stage(
            "evaluate_continued_sft_synthetic",
            lambda: evaluate_model(
                config,
                revision=model_revision,
                cache_dir=cache_dir,
                data_path=internal_path,
                output_dir=evaluation_root / "continued_sft" / "synthetic",
                adapter_path=continued_adapter,
            ),
        )
        external_metrics["continued_sft"] = stage(
            "evaluate_continued_sft_gsm8k",
            lambda: evaluate_model(
                external_config,
                revision=model_revision,
                cache_dir=cache_dir,
                data_path=external_path,
                output_dir=evaluation_root / "continued_sft" / "gsm8k",
                adapter_path=continued_adapter,
            ),
        )

        internal_samples = int(config["evaluation"]["bootstrap_samples"])
        internal_seed = int(config["evaluation"]["bootstrap_seed"])
        external_samples = int(config["external_evaluation"]["bootstrap_samples"])
        external_seed = int(config["external_evaluation"]["bootstrap_seed"])
        comparisons = stage(
            "paired_comparisons",
            lambda: {
                "internal_base_to_sft": _compare(
                    evaluation_root / "base" / "synthetic" / "predictions.jsonl",
                    evaluation_root / "sft" / "synthetic" / "predictions.jsonl",
                    bootstrap_samples=internal_samples,
                    bootstrap_seed=internal_seed,
                ),
                "internal_sft_to_continued_sft": _compare(
                    evaluation_root / "sft" / "synthetic" / "predictions.jsonl",
                    evaluation_root
                    / "continued_sft"
                    / "synthetic"
                    / "predictions.jsonl",
                    bootstrap_samples=internal_samples,
                    bootstrap_seed=internal_seed + 1,
                ),
                "external_base_to_sft": _compare(
                    evaluation_root / "base" / "gsm8k" / "predictions.jsonl",
                    evaluation_root / "sft" / "gsm8k" / "predictions.jsonl",
                    bootstrap_samples=external_samples,
                    bootstrap_seed=external_seed,
                ),
                "external_sft_to_continued_sft": _compare(
                    evaluation_root / "sft" / "gsm8k" / "predictions.jsonl",
                    evaluation_root / "continued_sft" / "gsm8k" / "predictions.jsonl",
                    bootstrap_samples=external_samples,
                    bootstrap_seed=external_seed + 1,
                ),
            },
        )
        write_json(destination / "comparisons.json", comparisons)

        readiness = stage(
            "readiness_gate",
            lambda: _readiness_gate(
                config,
                internal_metrics=internal_metrics,
                external_metrics=external_metrics,
                sft_summary=sft_summary,
                continued_sft_summary=continued_summary,
                model_revision=model_revision,
            ),
        )
        write_json(destination / "readiness.json", readiness)
        stage(
            "write_report",
            lambda: write_stage2_report(
                destination / "report.md",
                config=config,
                model_revision=model_revision,
                dataset_revision=dataset_revision,
                internal_metrics=internal_metrics,
                external_metrics=external_metrics,
                comparisons=comparisons,
                sft_summary=sft_summary,
                continued_sft_summary=continued_summary,
                readiness=readiness,
            ),
        )

        manifest["readiness"] = readiness
        manifest["artifacts"] = stage(
            "hash_primary_artifacts", lambda: _primary_artifact_hashes(destination)
        )
        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        write_json(manifest_path, manifest)
        (destination / "SUCCESS").write_text("completed\n", encoding="utf-8")
        return destination
    except Exception as error:
        manifest["status"] = "failed"
        manifest["finished_at"] = _utc_now()
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(manifest_path, manifest)
        raise


def refresh_stage2_audit(config: dict[str, Any], *, run_dir: str | Path) -> dict[str, Any]:
    """Recompute only derived readiness/report artifacts from an immutable completed run."""

    destination = Path(run_dir).resolve()
    manifest_path = destination / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed" or not (destination / "SUCCESS").is_file():
        raise ValueError("Only a completed Stage 2 run with SUCCESS can be audited")
    if manifest.get("config_hash") != config["_config_hash"]:
        raise ValueError("Audit config hash does not match the completed run")
    model_revision = str(manifest["model"]["resolved_revision"])
    dataset_revision = str(manifest["external_dataset"]["resolved_revision"])

    evaluation_root = destination / "evaluation"
    internal_metrics = {
        name: read_json(evaluation_root / name / "synthetic" / "metrics.json")
        for name in ("base", "sft", "continued_sft")
    }
    external_metrics = {
        name: read_json(evaluation_root / name / "gsm8k" / "metrics.json")
        for name in ("base", "sft", "continued_sft")
    }
    sft_summary = read_json(destination / "sft" / "training_summary.json")
    continued_summary = read_json(
        destination / "continued_sft" / "training_summary.json"
    )
    comparisons = read_json(destination / "comparisons.json")
    readiness = _readiness_gate(
        config,
        internal_metrics=internal_metrics,
        external_metrics=external_metrics,
        sft_summary=sft_summary,
        continued_sft_summary=continued_summary,
        model_revision=model_revision,
    )
    write_json(destination / "readiness.json", readiness)
    write_stage2_report(
        destination / "report.md",
        config=config,
        model_revision=model_revision,
        dataset_revision=dataset_revision,
        internal_metrics=internal_metrics,
        external_metrics=external_metrics,
        comparisons=comparisons,
        sft_summary=sft_summary,
        continued_sft_summary=continued_summary,
        readiness=readiness,
    )
    manifest["readiness"] = readiness
    manifest["artifacts"] = _primary_artifact_hashes(destination)
    manifest["post_run_audit"] = {
        "refreshed_at": _utc_now(),
        "config_hash": config["_config_hash"],
        "derived_artifacts_only": True,
    }
    write_json(manifest_path, manifest)
    return readiness


def mark_incomplete_run_aborted(
    *, run_dir: str | Path, reason: str
) -> dict[str, Any]:
    """Close an externally interrupted run without deleting its partial evidence."""

    destination = Path(run_dir).resolve()
    manifest_path = destination / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "running" or (destination / "SUCCESS").exists():
        raise ValueError("Only an incomplete running run can be marked aborted")
    if not reason.strip():
        raise ValueError("An abort reason is required")
    timestamp = _utc_now()
    for stage_info in manifest.get("stages", {}).values():
        if stage_info.get("status") == "running":
            stage_info["status"] = "aborted"
            stage_info["finished_at"] = timestamp
    manifest["status"] = "aborted"
    manifest["aborted_at"] = timestamp
    manifest["abort_reason"] = reason.strip()
    write_json(manifest_path, manifest)
    (destination / "ABORTED").write_text(reason.strip() + "\n", encoding="utf-8")
    return manifest
