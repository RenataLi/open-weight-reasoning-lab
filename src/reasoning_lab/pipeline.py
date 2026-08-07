from __future__ import annotations

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
from reasoning_lab.evaluation.generation import evaluate_model
from reasoning_lab.evaluation.metrics import compare_predictions
from reasoning_lab.io_utils import read_jsonl, sha256_file, write_json
from reasoning_lab.reporting.compare import write_report
from reasoning_lab.reproducibility import environment_snapshot, run_gpu_preflight
from reasoning_lab.training.sft import train_sft


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _new_run_dir(config: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return resolve_path(config, "runs_dir") / f"{timestamp}_{_safe_name(config['project']['run_name'])}"


def _resolve_model_revision(config: dict[str, Any]) -> str:
    info = HfApi().model_info(
        repo_id=config["model"]["id"],
        revision=config["model"]["revision"],
    )
    if not info.sha:
        raise RuntimeError("Hugging Face did not return an immutable model revision")
    return str(info.sha)


def run_pipeline(
    config: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
) -> Path:
    destination = Path(run_dir).resolve() if run_dir else _new_run_dir(config)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"A new run cannot overwrite an existing non-empty directory: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_path(config, "cache_dir")
    data_dir = resolve_path(config, "data_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)

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
    with (destination / "resolved_config.yaml").open("w", encoding="utf-8", newline="\n") as handle:
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
        preflight = stage("preflight", run_gpu_preflight)
        manifest["preflight"] = preflight

        revision = stage("resolve_model_revision", lambda: _resolve_model_revision(config))
        manifest["model"] = {
            "id": config["model"]["id"],
            "requested_revision": config["model"]["revision"],
            "resolved_revision": revision,
        }

        dataset_manifest = stage("generate_data", lambda: generate_dataset(config, data_dir))
        manifest["dataset"] = dataset_manifest
        split = str(config["evaluation"]["split"])
        test_path = data_dir / f"{split}.jsonl"

        baseline_dir = destination / "baseline"
        baseline_metrics = stage(
            "evaluate_base",
            lambda: evaluate_model(
                config,
                revision=revision,
                cache_dir=cache_dir,
                data_path=test_path,
                output_dir=baseline_dir,
            ),
        )

        training_dir = destination / "training"
        training_summary = stage(
            "train_sft",
            lambda: train_sft(
                config,
                revision=revision,
                cache_dir=cache_dir,
                train_path=data_dir / "train.jsonl",
                output_dir=training_dir,
                resume=False,
            ),
        )

        post_dir = destination / "post_sft"
        post_metrics = stage(
            "evaluate_post_sft",
            lambda: evaluate_model(
                config,
                revision=revision,
                cache_dir=cache_dir,
                data_path=test_path,
                output_dir=post_dir,
                adapter_path=training_dir / "final_adapter",
            ),
        )

        comparison = stage(
            "compare",
            lambda: compare_predictions(
                read_jsonl(baseline_dir / "predictions.jsonl"),
                read_jsonl(post_dir / "predictions.jsonl"),
                bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
                bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]),
            ),
        )
        write_json(destination / "comparison.json", comparison)
        write_report(
            destination / "report.md",
            config=config,
            revision=revision,
            baseline_metrics=baseline_metrics,
            post_metrics=post_metrics,
            comparison=comparison,
            training_summary=training_summary,
        )

        artifacts = {}
        for relative in (
            "environment.json",
            "resolved_config.yaml",
            "baseline/predictions.jsonl",
            "baseline/metrics.json",
            "training/training_summary.json",
            "post_sft/predictions.jsonl",
            "post_sft/metrics.json",
            "comparison.json",
            "report.md",
        ):
            path = destination / relative
            artifacts[relative] = sha256_file(path)
        manifest["artifacts"] = artifacts
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
