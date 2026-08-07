from __future__ import annotations

import copy
import hashlib
import json
import statistics
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch
import yaml

from reasoning_lab.config import public_config, resolve_path
from reasoning_lab.data.stage4_reasoning import generate_stage4_dataset
from reasoning_lab.data.svamp import prepare_svamp_snapshot
from reasoning_lab.evaluation.generation import evaluate_model
from reasoning_lab.evaluation.metrics import compare_predictions
from reasoning_lab.evaluation.preferences import evaluate_preference_adapter
from reasoning_lab.experiment_integrity import (
    attest_stage4_implementation,
    attest_stage3_source,
    claim_final_rows,
    collect_artifact_hashes,
    record_final_claim_completion,
    row_id_set_sha256,
    verify_artifact_hashes,
    verify_locked_file,
)
from reasoning_lab.io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from reasoning_lab.models.loading import load_tokenizer
from reasoning_lab.reporting.stage4 import (
    write_stage4_confirmation_report,
    write_stage4_search_report,
)
from reasoning_lab.reproducibility import environment_snapshot, run_gpu_preflight
from reasoning_lab.stage2_pipeline import _prepare_new_run_dir, _resolve_model_revision
from reasoning_lab.stage4_selection import select_control, select_dpo_candidate
from reasoning_lab.training.continued_sft import sha256_directory, train_continued_sft
from reasoning_lab.training.dpo import train_dpo_branch


RESEARCH_SOURCES = [
    {
        "title": "Direct Preference Optimization",
        "url": "https://arxiv.org/abs/2305.18290",
    },
    {
        "title": "TRL DPOTrainer documentation",
        "url": "https://huggingface.co/docs/trl/dpo_trainer",
    },
    {
        "title": "PEFT quantization guide",
        "url": "https://huggingface.co/docs/peft/developer_guides/quantization",
    },
    {
        "title": "SVAMP paper",
        "url": "https://aclanthology.org/2021.naacl-main.168/",
    },
    {
        "title": "Official SVAMP repository",
        "url": "https://github.com/arkilpatel/SVAMP",
    },
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_dir(config: dict[str, Any], phase: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = str(config["project"]["run_name"])
    return resolve_path(config, "runs_dir") / f"{timestamp}_{run_name}-{phase}"


def _write_resolved_config(config: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(public_config(config), handle, sort_keys=True, allow_unicode=True)


def _stage(
    manifest: dict[str, Any],
    manifest_path: Path,
    name: str,
    function: Callable[[], Any],
) -> Any:
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


def _absolute(config: dict[str, Any], raw: str | Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else Path(config["_project_root"]) / path).resolve()


def _parent_adapter(config: dict[str, Any]) -> Path:
    parent = _absolute(config, str(config["stage4"]["parent_adapter"]))
    if sha256_directory(parent) != str(config["stage4"]["parent_adapter_sha256"]):
        raise ValueError("Configured Stage 4 parent adapter hash mismatch")
    return parent


def _attest_stage4_predecessor(config: dict[str, Any]) -> dict[str, Any]:
    stage4 = config["stage4"]
    iteration = int(stage4.get("development_iteration", 1))
    if iteration <= 1:
        return {"development_iteration": iteration, "predecessor_present": False}
    predecessor = _absolute(config, str(stage4["predecessor_search"]))
    outcome_path = predecessor / "search_outcome.json"
    manifest_path = predecessor / "run_manifest.json"
    selection_path = predecessor / "selection_result.json"
    expected_outcome_hash = str(stage4["predecessor_search_outcome_sha256"])
    if sha256_file(outcome_path) != expected_outcome_hash:
        raise ValueError("Stage 4 predecessor search outcome hash mismatch")
    predecessor_outcome = read_json(outcome_path)
    predecessor_manifest = read_json(manifest_path)
    predecessor_selection = read_json(selection_path)
    if sha256_file(selection_path) != str(
        predecessor_outcome["selection_result_sha256"]
    ):
        raise ValueError("Stage 4 predecessor selection differs from its outcome")
    verify_artifact_hashes(
        predecessor, predecessor_outcome["search_evidence_sha256"]
    )
    if (
        predecessor_manifest.get("phase") != "stage4_search"
        or predecessor_manifest.get("status") != "completed_no_eligible_candidate"
        or predecessor_selection.get("selected_candidate") is not None
    ):
        raise ValueError("Stage 4 predecessor is not a completed no-eligible search")
    return {
        "development_iteration": iteration,
        "predecessor_present": True,
        "predecessor_search": str(predecessor),
        "predecessor_manifest_sha256": sha256_file(manifest_path),
        "predecessor_selection_sha256": sha256_file(selection_path),
        "predecessor_search_outcome_sha256": expected_outcome_hash,
        "adaptation_note": str(stage4["adaptation_note"]),
    }


def _evaluation_config(
    config: dict[str, Any], *, count: int, batch_size: int
) -> dict[str, Any]:
    derived = copy.deepcopy(config)
    derived["evaluation"]["max_examples"] = count
    derived["evaluation"]["batch_size"] = batch_size
    return derived


def _locked_role(
    *,
    role: str,
    path: Path,
    metadata: dict[str, Any],
    source_dataset: str,
    source_revision: str,
) -> dict[str, Any]:
    return {
        **metadata,
        "role": role,
        "path": str(path.resolve()),
        "source_dataset": source_dataset,
        "source_revision": source_revision,
    }


def _prior_stage4_development_exposure(
    config: dict[str, Any], *, current_run: Path
) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    """Collect prior Stage 4 development material that may not become fresh final data."""

    runs_root = resolve_path(config, "runs_dir").resolve()
    reserved_reasoning_prompts: set[str] = set()
    forced_svamp_validation_ids: set[str] = set()
    exposed_svamp_ids: set[str] = set()
    attempts: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        run_root = manifest_path.parent.resolve()
        if run_root == current_run.resolve():
            continue
        try:
            run_manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if run_manifest.get("phase") != "stage4_search":
            continue
        data_manifest_path = run_root / "frozen_data" / "stage4_data_manifest.json"
        if not data_manifest_path.is_file():
            continue
        data_manifest = read_json(data_manifest_path)
        role_records: dict[str, Any] = {}
        for role, item in data_manifest.get("roles", {}).items():
            path = Path(str(item.get("path", "")))
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            role_records[str(role)] = {
                "count": len(rows),
                "sha256": sha256_file(path),
            }
            if str(item.get("source_dataset")) == "stage4_reasoning_synthetic":
                reserved_reasoning_prompts.update(
                    str(row["prompt"]).strip()
                    for row in rows
                    if row.get("prompt")
                )
            if str(item.get("source_dataset")) == "arkilpatel/SVAMP":
                exposed_svamp_ids.update(
                    str(row.get("source_id") or row.get("id")) for row in rows
                )
            if str(role) == "stage4_svamp_validation":
                forced_svamp_validation_ids.update(
                    str(row.get("source_id") or row.get("id")) for row in rows
                )
        attempts.append(
            {
                "run_dir": str(run_root),
                "status": str(run_manifest.get("status")),
                "config_hash": str(run_manifest.get("config_hash")),
                "data_manifest_sha256": sha256_file(data_manifest_path),
                "roles": role_records,
            }
        )
    inventory = {
        "policy": (
            "Prior synthetic prompts are reserved from every new role. Prior SVAMP "
            "validation IDs remain fixed, and every other valid SVAMP row stays in the "
            "development quarantine; SVAMP has no confirmation role."
        ),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "reserved_reasoning_prompt_count": len(reserved_reasoning_prompts),
        "forced_svamp_validation_id_count": len(forced_svamp_validation_ids),
        "exposed_svamp_id_count": len(exposed_svamp_ids),
        "exposed_svamp_id_set_sha256": row_id_set_sha256(
            [{"id": value} for value in sorted(exposed_svamp_ids)]
        ),
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                attempts,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    return (
        reserved_reasoning_prompts,
        forced_svamp_validation_ids,
        exposed_svamp_ids,
        inventory,
    )


def _prepare_stage4_data(
    config: dict[str, Any],
    *,
    revision: str,
    run_root: Path,
) -> dict[str, Any]:
    stage4 = config["stage4"]
    frozen = run_root / "frozen_data"
    reasoning_root = frozen / "reasoning"
    svamp_root = frozen / "svamp"
    (
        reserved_reasoning_prompts,
        forced_svamp_validation_ids,
        exposed_svamp_ids,
        prior_stage4_inventory,
    ) = _prior_stage4_development_exposure(config, current_run=run_root)
    tokenizer = load_tokenizer(
        config,
        revision,
        resolve_path(config, "cache_dir"),
        padding_side="right",
    )
    reasoning_config = {
        "preference_train_size": int(stage4["preference_train_size"]),
        "preference_validation_size": int(stage4["preference_validation_size"]),
        "preference_confirmation_size": int(stage4["preference_confirmation_size"]),
        "benchmark_validation_size": int(stage4["synthetic_validation_size"]),
        "benchmark_confirmation_size": int(stage4["synthetic_confirmation_size"]),
        "seed": int(stage4["data_seed"]),
        "generator_version": str(stage4["data_version"]),
        "max_pair_token_delta": 2,
        "max_completion_tokens": 256,
    }
    reasoning_manifest = generate_stage4_dataset(
        reasoning_config,
        reasoning_root,
        tokenizer=tokenizer,
        reserved_prompts=reserved_reasoning_prompts,
    )
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    svamp_manifest = prepare_svamp_snapshot(
        config,
        source_path=resolve_path(config, "svamp_source"),
        output_dir=svamp_root,
        forced_validation_ids=forced_svamp_validation_ids or None,
    )
    role_mapping = {
        "preference_train": "stage4_preference_train",
        "preference_validation": "stage4_preference_validation",
        "preference_confirmation": "stage4_preference_confirmation",
        "benchmark_validation": "stage4_synthetic_validation",
        "benchmark_confirmation": "stage4_synthetic_confirmation",
    }
    roles: dict[str, Any] = {}
    for source_role, final_role in role_mapping.items():
        metadata = reasoning_manifest["roles"][source_role]
        roles[final_role] = _locked_role(
            role=final_role,
            path=reasoning_root / f"{source_role}.jsonl",
            metadata=metadata,
            source_dataset="stage4_reasoning_synthetic",
            source_revision=str(stage4["data_version"]),
        )
    svamp_filenames = {
        "stage4_svamp_validation": "validation.jsonl",
        "stage4_svamp_development_quarantine": "development_quarantine.jsonl",
    }
    if set(svamp_manifest["files"]) != set(svamp_filenames):
        raise ValueError("Unexpected SVAMP role inventory")
    for role, filename in svamp_filenames.items():
        metadata = svamp_manifest["files"][role]
        roles[role] = _locked_role(
            role=role,
            path=svamp_root / filename,
            metadata=metadata,
            source_dataset="arkilpatel/SVAMP",
            source_revision=str(svamp_manifest["source_commit"]),
        )

    preference_train = read_jsonl(Path(roles["stage4_preference_train"]["path"]))
    chosen_rows = [
        {
            "id": str(row["id"]),
            "family": str(row["family"]),
            "difficulty": int(row["difficulty"]),
            "prompt": str(row["prompt"]),
            "completion": str(row["chosen"]),
            "source_pair_sha256": hashlib.sha256(
                (str(row["prompt"]) + "\0" + str(row["chosen"])).encode("utf-8")
            ).hexdigest(),
        }
        for row in preference_train
    ]
    chosen_path = frozen / "chosen_sft_train.jsonl"
    write_jsonl(chosen_path, chosen_rows)
    chosen_manifest = {
        "path": str(chosen_path.resolve()),
        "count": len(chosen_rows),
        "sha256": sha256_file(chosen_path),
        "row_id_set_sha256": row_id_set_sha256(chosen_rows),
        "derived_one_to_one_from": roles["stage4_preference_train"]["sha256"],
        "same_id_order": [row["id"] for row in preference_train]
        == [row["id"] for row in chosen_rows],
        "compute_matching_note": (
            "Controls are prompt/example/update matched; DPO additionally processes rejected "
            "and reference completions, so this is not a FLOP-matched comparison."
        ),
    }
    write_json(frozen / "chosen_sft_train.manifest.json", chosen_manifest)

    historical_files = sorted((Path(config["_project_root"]) / "data").rglob("*.jsonl"))
    stage3_frozen = _absolute(config, stage4["source_stage3_search"]) / "frozen_data"
    historical_files.extend(sorted(stage3_frozen.glob("*.jsonl")))
    historical_ids: set[str] = set()
    historical_prompts: set[str] = set()
    inventory_records: list[dict[str, Any]] = []
    for path in dict.fromkeys(candidate.resolve() for candidate in historical_files):
        rows = read_jsonl(path)
        historical_ids.update(str(row["id"]) for row in rows if row.get("id") is not None)
        historical_prompts.update(
            str(row["prompt"]).strip() for row in rows if row.get("prompt")
        )
        inventory_records.append(
            {"path": str(path), "sha256": sha256_file(path), "count": len(rows)}
        )
    stage4_rows = [
        row
        for role, item in roles.items()
        if item["source_dataset"] == "stage4_reasoning_synthetic"
        for row in read_jsonl(Path(item["path"]))
    ]
    all_new_rows = [
        row for item in roles.values() for row in read_jsonl(Path(item["path"]))
    ]
    id_overlap = sorted({str(row["id"]) for row in all_new_rows} & historical_ids)
    prompt_overlap = sorted(
        {str(row["prompt"]).strip() for row in stage4_rows} & historical_prompts
    )
    stage4_prompt_values = [str(row["prompt"]).strip() for row in stage4_rows]
    if len(stage4_prompt_values) != len(set(stage4_prompt_values)):
        raise ValueError("Stage 4 synthetic roles contain duplicate exact prompts")
    reserved_overlap = sorted(
        set(stage4_prompt_values) & reserved_reasoning_prompts
    )
    frozen_svamp_ids = {
        str(row.get("source_id") or row.get("id"))
        for role, item in roles.items()
        if item["source_dataset"] == "arkilpatel/SVAMP"
        for row in read_jsonl(Path(item["path"]))
    }
    unquarantined_prior_svamp = sorted(exposed_svamp_ids - frozen_svamp_ids)
    all_new_ids = [str(row["id"]) for row in all_new_rows]
    if len(all_new_ids) != len(set(all_new_ids)):
        raise ValueError("Stage 4 frozen roles contain duplicate row IDs")
    if id_overlap or prompt_overlap or reserved_overlap or unquarantined_prior_svamp:
        raise ValueError("Stage 4 data overlaps historical IDs or exact prompts")
    historical_audit = {
        "historical_file_count": len(inventory_records),
        "historical_row_id_count": len(historical_ids),
        "historical_prompt_count": len(historical_prompts),
        "historical_inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "stage4_id_overlap_count": 0,
        "stage4_exact_prompt_overlap_count": 0,
        "stage4_internal_exact_prompt_overlap_count": 0,
        "prior_stage4_reserved_prompt_overlap_count": 0,
        "prior_exposed_svamp_unquarantined_count": 0,
        "svamp_final_role_present": False,
    }
    manifest = {
        "protocol": str(stage4["protocol_id"]),
        "created_at": _utc_now(),
        "roles": roles,
        "chosen_sft_train": chosen_manifest,
        "reasoning_manifest": reasoning_manifest,
        "reasoning_manifest_sha256": sha256_file(reasoning_root / "dataset_manifest.json"),
        "svamp_manifest": svamp_manifest,
        "svamp_manifest_sha256": sha256_file(svamp_root / "dataset_manifest.json"),
        "historical_leakage_audit": historical_audit,
        "prior_stage4_development_exposure": prior_stage4_inventory,
    }
    write_json(frozen / "stage4_data_manifest.json", manifest)
    return manifest


def _control_config(
    config: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    derived = copy.deepcopy(config)
    derived["continuation"]["max_steps"] = int(control["max_steps"])
    derived["continuation"]["seed"] = int(config["stage4"]["dpo_training"]["seed"])
    derived["continuation"]["data_seed"] = int(
        config["stage4"]["dpo_training"]["data_seed"]
    )
    derived["continuation"]["learning_rate"] = float(
        config["stage4"]["dpo_training"]["learning_rate"]
    )
    derived["continuation"]["max_length"] = int(
        config["stage4"]["dpo_training"]["max_length"]
    )
    derived["continuation"]["disable_dropout"] = True
    derived["continuation"]["save_strategy"] = "no"
    return derived


def _evaluate_generation_pair(
    config: dict[str, Any],
    *,
    revision: str,
    cache_dir: Path,
    data_manifest: dict[str, Any],
    adapter: Path,
    output_root: Path,
) -> dict[str, Any]:
    stage4 = config["stage4"]
    synthetic = data_manifest["roles"]["stage4_synthetic_validation"]
    external = data_manifest["roles"]["stage4_svamp_validation"]
    synthetic_metrics = evaluate_model(
        _evaluation_config(
            config,
            count=int(synthetic["count"]),
            batch_size=int(stage4["synthetic_batch_size"]),
        ),
        revision=revision,
        cache_dir=cache_dir,
        data_path=synthetic["path"],
        output_dir=output_root / "synthetic",
        adapter_path=adapter,
        evaluation_role="stage4_synthetic_validation",
    )
    external_metrics = evaluate_model(
        _evaluation_config(
            config,
            count=int(external["count"]),
            batch_size=int(stage4["svamp_batch_size"]),
        ),
        revision=revision,
        cache_dir=cache_dir,
        data_path=external["path"],
        output_dir=output_root / "svamp",
        adapter_path=adapter,
        evaluation_role="stage4_svamp_validation",
    )
    return {"synthetic": synthetic_metrics, "external": external_metrics}


def _headroom_gate(base: dict[str, Any]) -> dict[str, Any]:
    exact = float(base["synthetic"]["overall"]["answer_exact_match"])
    family_values = [
        float(group["answer_exact_match"])
        for group in base["synthetic"]["by_family"].values()
    ]
    checks = {
        "overall_headroom": 0.10 <= exact <= 0.95,
        "multiple_non_ceiling_families": sum(value < 0.95 for value in family_values) >= 3,
        "parseable": float(base["synthetic"]["overall"]["parse_success_rate"]) >= 0.99,
    }
    return {"passed": all(checks.values()), "checks": checks, "synthetic_exact": exact}


def _search_evidence_paths(
    run_root: Path,
    *,
    candidate_ids: list[str],
    control_ids: list[str],
) -> list[Path]:
    paths = [
        run_root / "resolved_config.yaml",
        run_root / "source_attestation.json",
        run_root / "implementation_attestation.json",
        run_root / "predecessor_attestation.json",
        run_root / "environment.json",
        run_root / "frozen_data" / "stage4_data_manifest.json",
        run_root / "frozen_data" / "reasoning" / "dataset_manifest.json",
        run_root / "frozen_data" / "reasoning" / "preference_train.jsonl",
        run_root / "frozen_data" / "reasoning" / "preference_validation.jsonl",
        run_root / "frozen_data" / "reasoning" / "benchmark_validation.jsonl",
        run_root / "frozen_data" / "svamp" / "dataset_manifest.json",
        run_root / "frozen_data" / "svamp" / "validation.jsonl",
        run_root
        / "frozen_data"
        / "svamp"
        / "development_quarantine.jsonl",
        run_root / "frozen_data" / "chosen_sft_train.jsonl",
        run_root / "frozen_data" / "chosen_sft_train.manifest.json",
        run_root / "development_metrics.json",
        run_root / "training_summaries.json",
        run_root / "selection_result.json",
        run_root / "control_selection.json",
        run_root / "report.md",
        run_root / "evaluation" / "parent" / "synthetic" / "metrics.json",
        run_root / "evaluation" / "parent" / "synthetic" / "predictions.jsonl",
        run_root / "evaluation" / "parent" / "svamp" / "metrics.json",
        run_root / "evaluation" / "parent" / "svamp" / "predictions.jsonl",
    ]
    for candidate_id in candidate_ids:
        paths.extend(
            [
                run_root / "candidates" / candidate_id / "dpo" / "training_summary.json",
                run_root / "evaluation" / candidate_id / "synthetic" / "metrics.json",
                run_root / "evaluation" / candidate_id / "synthetic" / "predictions.jsonl",
                run_root / "evaluation" / candidate_id / "svamp" / "metrics.json",
                run_root / "evaluation" / candidate_id / "svamp" / "predictions.jsonl",
                run_root / "evaluation" / candidate_id / "preferences" / "metrics.json",
                run_root
                / "evaluation"
                / candidate_id
                / "preferences"
                / "preference_scores.jsonl",
            ]
        )
    for control_id in control_ids:
        paths.extend(
            [
                run_root / "controls" / control_id / "sft" / "training_summary.json",
                run_root / "evaluation" / control_id / "synthetic" / "metrics.json",
                run_root / "evaluation" / control_id / "synthetic" / "predictions.jsonl",
                run_root / "evaluation" / control_id / "svamp" / "metrics.json",
                run_root / "evaluation" / control_id / "svamp" / "predictions.jsonl",
                run_root / "evaluation" / control_id / "preferences" / "metrics.json",
                run_root
                / "evaluation"
                / control_id
                / "preferences"
                / "preference_scores.jsonl",
            ]
        )
    return paths


def run_stage4_search(
    config: dict[str, Any], *, run_dir: str | Path | None = None
) -> Path:
    """Run development-only DPO/control search and lock one candidate."""

    destination = _prepare_new_run_dir(run_dir or _new_run_dir(config, "search"))
    manifest_path = destination / "run_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "phase": "stage4_search",
        "started_at": _utc_now(),
        "run_dir": str(destination),
        "config_hash": config["_config_hash"],
        "test_metrics_used_for_selection": False,
        "stages": {},
    }
    write_json(manifest_path, manifest)
    _write_resolved_config(config, destination / "resolved_config.yaml")
    try:
        attestation = _stage(
            manifest,
            manifest_path,
            "attest_stage3_source",
            lambda: attest_stage3_source(config),
        )
        write_json(destination / "source_attestation.json", attestation)
        predecessor_attestation = _stage(
            manifest,
            manifest_path,
            "attest_stage4_predecessor",
            lambda: _attest_stage4_predecessor(config),
        )
        write_json(
            destination / "predecessor_attestation.json",
            predecessor_attestation,
        )
        implementation_attestation = _stage(
            manifest,
            manifest_path,
            "attest_stage4_implementation",
            lambda: attest_stage4_implementation(config),
        )
        write_json(
            destination / "implementation_attestation.json",
            implementation_attestation,
        )
        revision = _stage(
            manifest,
            manifest_path,
            "resolve_model_revision",
            lambda: _resolve_model_revision(config),
        )
        environment = _stage(
            manifest,
            manifest_path,
            "environment",
            environment_snapshot,
        )
        preflight = _stage(
            manifest,
            manifest_path,
            "gpu_preflight",
            run_gpu_preflight,
        )
        write_json(
            destination / "environment.json",
            {"environment": environment, "preflight": preflight},
        )
        data_manifest = _stage(
            manifest,
            manifest_path,
            "freeze_stage4_data",
            lambda: _prepare_stage4_data(
                config, revision=revision, run_root=destination
            ),
        )
        parent = _parent_adapter(config)
        cache_dir = resolve_path(config, "cache_dir")
        base = _stage(
            manifest,
            manifest_path,
            "evaluate_parent_validation",
            lambda: _evaluate_generation_pair(
                config,
                revision=revision,
                cache_dir=cache_dir,
                data_manifest=data_manifest,
                adapter=parent,
                output_root=destination / "evaluation" / "parent",
            ),
        )
        headroom = _headroom_gate(base)
        write_json(destination / "headroom_gate.json", headroom)
        if not headroom["passed"]:
            raise RuntimeError(
                "Stage 4 synthetic development set failed the preregistered headroom gate"
            )

        controls: list[dict[str, Any]] = []
        control_summaries: dict[str, Any] = {}
        chosen_path = Path(data_manifest["chosen_sft_train"]["path"])
        for control in config["stage4"]["sft_controls"]:
            control_id = str(control["id"])
            control_config = _control_config(config, control)
            summary = _stage(
                manifest,
                manifest_path,
                f"train_{control_id}",
                lambda control_config=control_config, control_id=control_id: train_continued_sft(
                    control_config,
                    revision=revision,
                    cache_dir=cache_dir,
                    continuation_path=chosen_path,
                    parent_adapter=parent,
                    output_dir=destination / "controls" / control_id / "sft",
                ),
            )
            control_summaries[control_id] = summary
            adapter = Path(summary["final_adapter"])
            generation = _stage(
                manifest,
                manifest_path,
                f"evaluate_{control_id}_validation",
                lambda adapter=adapter, control_id=control_id: _evaluate_generation_pair(
                    config,
                    revision=revision,
                    cache_dir=cache_dir,
                    data_manifest=data_manifest,
                    adapter=adapter,
                    output_root=destination / "evaluation" / control_id,
                ),
            )
            control_preference = _stage(
                manifest,
                manifest_path,
                f"score_{control_id}_preferences",
                lambda adapter=adapter, control_id=control_id: evaluate_preference_adapter(
                    config,
                    revision=revision,
                    cache_dir=cache_dir,
                    data_path=Path(
                        data_manifest["roles"]["stage4_preference_validation"]["path"]
                    ),
                    output_dir=destination
                    / "evaluation"
                    / control_id
                    / "preferences",
                    policy_adapter=adapter,
                    reference_adapter=parent,
                    beta=float(config["stage4"]["dpo_training"]["beta"]),
                    loss_type="sigmoid",
                    evaluation_role="stage4_preference_validation",
                ),
            )
            controls.append(
                {"id": control_id, **generation, "preference": control_preference}
            )

        candidates: list[dict[str, Any]] = []
        candidate_summaries: dict[str, Any] = {}
        preference_train = Path(
            data_manifest["roles"]["stage4_preference_train"]["path"]
        )
        preference_validation = Path(
            data_manifest["roles"]["stage4_preference_validation"]["path"]
        )
        for candidate in config["stage4"]["candidates"]:
            candidate_id = str(candidate["id"])
            summary = _stage(
                manifest,
                manifest_path,
                f"train_{candidate_id}",
                lambda candidate=candidate, candidate_id=candidate_id: train_dpo_branch(
                    config,
                    candidate=candidate,
                    revision=revision,
                    cache_dir=cache_dir,
                    train_path=preference_train,
                    validation_path=preference_validation,
                    parent_adapter=parent,
                    output_dir=destination / "candidates" / candidate_id / "dpo",
                ),
            )
            candidate_summaries[candidate_id] = summary
            adapter = Path(summary["final_adapter"])
            generation = _stage(
                manifest,
                manifest_path,
                f"evaluate_{candidate_id}_validation",
                lambda adapter=adapter, candidate_id=candidate_id: _evaluate_generation_pair(
                    config,
                    revision=revision,
                    cache_dir=cache_dir,
                    data_manifest=data_manifest,
                    adapter=adapter,
                    output_root=destination / "evaluation" / candidate_id,
                ),
            )
            preference = _stage(
                manifest,
                manifest_path,
                f"score_{candidate_id}_preferences",
                lambda adapter=adapter, candidate=candidate, candidate_id=candidate_id: evaluate_preference_adapter(
                    config,
                    revision=revision,
                    cache_dir=cache_dir,
                    data_path=preference_validation,
                    output_dir=destination
                    / "evaluation"
                    / candidate_id
                    / "preferences",
                    policy_adapter=adapter,
                    reference_adapter=parent,
                    beta=float(candidate["beta"]),
                    loss_type=str(candidate["loss_type"]),
                    evaluation_role="stage4_preference_validation",
                ),
            )
            candidates.append(
                {"id": candidate_id, **generation, "preference": preference}
            )

        write_json(
            destination / "training_summaries.json",
            {"candidates": candidate_summaries, "controls": control_summaries},
        )
        development_metrics = {
            "base": base,
            "candidates": candidates,
            "controls": controls,
            "input_roles": [
                "stage4_synthetic_validation",
                "stage4_svamp_validation",
                "stage4_preference_validation",
            ],
            "test_metrics_used": False,
        }
        write_json(destination / "development_metrics.json", development_metrics)
        candidate_ids = [str(candidate["id"]) for candidate in config["stage4"]["candidates"]]
        control_ids = [str(control["id"]) for control in config["stage4"]["sft_controls"]]
        selection = select_dpo_candidate(
            base_synthetic=base["synthetic"],
            base_external=base["external"],
            candidates=candidates,
            expected_candidate_ids=candidate_ids,
            policy=config["stage4"]["selection_policy"],
        )
        control_selection = select_control(controls, expected_ids=control_ids)
        write_json(destination / "selection_result.json", selection)
        write_json(destination / "control_selection.json", control_selection)
        write_stage4_search_report(
            destination / "report.md",
            selection=selection,
            control_selection=control_selection,
            base=base,
            candidates=candidates,
            controls=controls,
            data_manifest=data_manifest,
        )
        if selection["selected_candidate"] is None:
            evidence = collect_artifact_hashes(
                destination,
                _search_evidence_paths(
                    destination,
                    candidate_ids=candidate_ids,
                    control_ids=control_ids,
                ),
            )
            outcome = {
                "status": "completed_no_eligible_candidate",
                "finished_at": _utc_now(),
                "config_hash": config["_config_hash"],
                "selection_result_sha256": sha256_file(
                    destination / "selection_result.json"
                ),
                "control_selection_sha256": sha256_file(
                    destination / "control_selection.json"
                ),
                "data_manifest_sha256": sha256_file(
                    destination / "frozen_data" / "stage4_data_manifest.json"
                ),
                "search_evidence_sha256": evidence,
                "confirmation_permitted": False,
            }
            write_json(destination / "search_outcome.json", outcome)
            manifest.update(
                {
                    "status": "completed_no_eligible_candidate",
                    "finished_at": _utc_now(),
                    "selection": selection,
                    "primary_artifacts": collect_artifact_hashes(
                        destination,
                        [
                            "search_outcome.json",
                            "selection_result.json",
                            "control_selection.json",
                            "development_metrics.json",
                            "report.md",
                        ],
                    ),
                }
            )
            write_json(manifest_path, manifest)
            (destination / "SEARCH_NO_ELIGIBLE_CANDIDATE").write_text(
                "No candidate passed the preregistered development gate.\n",
                encoding="utf-8",
            )
            return destination

        selected_id = str(selection["selected_candidate"])
        selected_control_id = str(control_selection["selected_control"])
        evidence = collect_artifact_hashes(
            destination,
            _search_evidence_paths(
                destination,
                candidate_ids=candidate_ids,
                control_ids=control_ids,
            ),
        )
        selected_candidate_spec = next(
            item for item in config["stage4"]["candidates"] if item["id"] == selected_id
        )
        selected_control_spec = next(
            item
            for item in config["stage4"]["sft_controls"]
            if item["id"] == selected_control_id
        )
        selected_summary = candidate_summaries[selected_id]
        selected_control_summary = control_summaries[selected_control_id]
        confirmation_roles = (
            "stage4_synthetic_confirmation",
            "stage4_preference_confirmation",
        )
        lock = {
            "protocol_id": str(config["stage4"]["protocol_id"]),
            "selection_locked_at": _utc_now(),
            "config_hash": config["_config_hash"],
            "model_revision": revision,
            "test_metrics_used_for_selection": False,
            "selection_input_roles": selection["selection_input_roles"],
            "selection_result_sha256": sha256_file(destination / "selection_result.json"),
            "control_selection_sha256": sha256_file(
                destination / "control_selection.json"
            ),
            "source_attestation_sha256": sha256_file(
                destination / "source_attestation.json"
            ),
            "predecessor_attestation_sha256": sha256_file(
                destination / "predecessor_attestation.json"
            ),
            "implementation_attestation_sha256": sha256_file(
                destination / "implementation_attestation.json"
            ),
            "implementation_combined_sha256": implementation_attestation[
                "combined_sha256"
            ],
            "search_environment_sha256": sha256_file(
                destination / "environment.json"
            ),
            "parent": {
                "adapter": str(parent),
                "adapter_sha256": sha256_directory(parent),
                "selection_rule": str(config["stage4"]["parent_selection_rule"]),
                "seed": int(config["stage4"]["parent_seed"]),
            },
            "selected_candidate": {
                "id": selected_id,
                "spec": selected_candidate_spec,
                "adapter": selected_summary["final_adapter"],
                "adapter_sha256": selected_summary["final_adapter_sha256"],
                "training_summary_sha256": sha256_file(
                    destination
                    / "candidates"
                    / selected_id
                    / "dpo"
                    / "training_summary.json"
                ),
            },
            "selected_control": {
                "id": selected_control_id,
                "spec": selected_control_spec,
                "adapter": selected_control_summary["final_adapter"],
                "adapter_sha256": selected_control_summary["final_adapter_sha256"],
                "training_summary_sha256": sha256_file(
                    destination
                    / "controls"
                    / selected_control_id
                    / "sft"
                    / "training_summary.json"
                ),
            },
            "confirmation_files": {
                role: data_manifest["roles"][role] for role in confirmation_roles
            },
            "data_manifest_sha256": sha256_file(
                destination / "frozen_data" / "stage4_data_manifest.json"
            ),
            "search_evidence_sha256": evidence,
            "research_sources": RESEARCH_SOURCES,
        }
        write_json(destination / "selection_lock.json", lock)
        selection_lock_hash = sha256_file(destination / "selection_lock.json")
        (destination / "SELECTION_LOCKED").write_text(
            selection_lock_hash + "\n", encoding="utf-8"
        )
        (destination / "SEARCH_SUCCESS").write_text("completed\n", encoding="utf-8")
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "model_revision": revision,
                "selection": {
                    "candidate": selected_id,
                    "control": selected_control_id,
                },
                "selection_lock_sha256": selection_lock_hash,
                "primary_artifacts": collect_artifact_hashes(
                    destination,
                    [
                        "selection_lock.json",
                        "selection_result.json",
                        "control_selection.json",
                        "development_metrics.json",
                        "report.md",
                        "SELECTION_LOCKED",
                    ],
                ),
            }
        )
        write_json(manifest_path, manifest)
        return destination
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(),
            }
        )
        write_json(manifest_path, manifest)
        (destination / "FAILED").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise


def _compare_paths(
    left: Path, right: Path, *, samples: int, seed: int
) -> dict[str, Any]:
    return compare_predictions(
        read_jsonl(left),
        read_jsonl(right),
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )


def _confirmation_evaluation_config(
    config: dict[str, Any], *, role: str, count: int
) -> dict[str, Any]:
    if role == "stage4_synthetic_confirmation":
        batch_size = int(config["stage4"]["synthetic_batch_size"])
    else:
        raise ValueError(f"Unknown generation confirmation role: {role}")
    return _evaluation_config(config, count=count, batch_size=batch_size)


def _evaluate_confirmation_branch(
    config: dict[str, Any],
    *,
    revision: str,
    cache_dir: Path,
    files: dict[str, Any],
    adapter: Path,
    output_root: Path,
) -> dict[str, Any]:
    synthetic = files["stage4_synthetic_confirmation"]
    synthetic_metrics = evaluate_model(
        _confirmation_evaluation_config(
            config,
            role="stage4_synthetic_confirmation",
            count=int(synthetic["count"]),
        ),
        revision=revision,
        cache_dir=cache_dir,
        data_path=synthetic["path"],
        output_dir=output_root / "synthetic",
        adapter_path=adapter,
        evaluation_role="stage4_synthetic_confirmation",
    )
    return {"synthetic": synthetic_metrics}


def _confirmation_verdict(
    config: dict[str, Any],
    *,
    metrics: dict[str, Any],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    policy = config["stage4"]["confirmation_policy"]
    parent_to_dpo = comparisons["parent_to_dpo"]
    control_to_dpo = comparisons["control_to_dpo"]
    dpo_synthetic = metrics["dpo"]["synthetic"]["overall"]
    parent_synthetic = metrics["parent"]["synthetic"]["overall"]
    synthetic_length_ratio = float(dpo_synthetic["generated_tokens_mean"]) / float(
        parent_synthetic["generated_tokens_mean"]
    )
    preference = metrics["dpo_preference"]["objective"]["overall"]
    control_preference = metrics["control_preference"]["objective"]["overall"]
    safety_checks = {
        "synthetic_point_retention": parent_to_dpo["synthetic"][
            "answer_exact_match_delta"
        ]
        >= -float(policy["max_synthetic_regression"]),
        "synthetic_ci_retention": parent_to_dpo["synthetic"][
            "paired_bootstrap_95_ci"
        ][0]
        > -float(policy["max_synthetic_ci_regression"]),
        "worst_family_retention": parent_to_dpo["synthetic"][
            "worst_family_delta"
        ]
        >= -float(policy["max_family_regression"]),
        "synthetic_format": float(dpo_synthetic["format_valid_rate"])
        >= max(
            float(policy["min_synthetic_format_valid_rate"]),
            float(parent_synthetic["format_valid_rate"])
            - float(policy["max_synthetic_format_regression"]),
        ),
        "synthetic_strict_retention": float(dpo_synthetic["strict_correct_rate"])
        >= float(parent_synthetic["strict_correct_rate"])
        - float(policy["max_synthetic_strict_regression"]),
        "synthetic_parse": float(dpo_synthetic["parse_success_rate"])
        >= float(policy["min_parse_success_rate"]),
        "synthetic_token_limit": float(dpo_synthetic["hit_max_new_tokens_rate"])
        <= float(policy["max_synthetic_token_limit_rate"]),
        "synthetic_length": float(policy["min_length_ratio"])
        <= synthetic_length_ratio
        <= float(policy["max_length_ratio"]),
        "heldout_preference_accuracy": float(preference["reward_accuracy"])
        >= float(policy["min_preference_accuracy"]),
        "heldout_preference_margin": float(preference["reward_margin_mean"]) > 0.0,
    }
    safe = all(safety_checks.values())
    benefit_delta = float(policy["objective_benefit_delta"])
    synthetic_benefit = (
        float(control_to_dpo["synthetic"]["answer_exact_match_delta"])
        >= benefit_delta
        and float(control_to_dpo["synthetic"]["paired_bootstrap_95_ci"][0]) > 0.0
    )
    preference_advantage = (
        float(preference["reward_accuracy"])
        - float(control_preference["reward_accuracy"])
        >= float(policy["min_preference_accuracy_advantage_vs_control"])
        and float(preference["reward_margin_mean"])
        > float(control_preference["reward_margin_mean"])
    )
    primary_endpoint = str(config["stage4"]["primary_benefit_endpoint"])
    if primary_endpoint != "synthetic":
        raise ValueError("Stage 4 currently preregisters synthetic as its sole primary endpoint")
    benefit = safe and synthetic_benefit and preference_advantage
    if not safe:
        status = "DPO PILOT FAILED SAFETY GATE"
        interpretation = (
            "The workflow executed, but the locked DPO branch violated at least one "
            "pre-registered retention, format, or preference gate. No capability benefit "
            "should be claimed."
        )
    elif benefit:
        status = "OBJECTIVE-SPECIFIC BENEFIT SUPPORTED"
        interpretation = (
            "The locked DPO branch was safe relative to its frozen parent and exceeded the "
            "strongest selected extra-SFT control on the preregistered synthetic primary "
            "endpoint and held-out preference shift. This supports a scoped "
            "objective-specific benefit on this pilot."
        )
    else:
        status = "DPO PIPELINE VALIDATED; BENEFIT INCONCLUSIVE"
        interpretation = (
            "The DPO implementation and held-out preference shift were validated without a "
            "disqualifying capability regression, but DPO did not beat the strongest extra-SFT "
            "control with the required paired evidence."
        )
    return {
        "status": status,
        "safe": safe,
        "objective_specific_benefit": benefit,
        "safety_checks": safety_checks,
        "benefit_checks": {
            "primary_endpoint": primary_endpoint,
            "synthetic_benefit": synthetic_benefit,
            "preference_advantage_vs_control": preference_advantage,
            "preference_accuracy_delta_vs_control": float(
                preference["reward_accuracy"]
            )
            - float(control_preference["reward_accuracy"]),
            "preference_margin_delta_vs_control": float(
                preference["reward_margin_mean"]
            )
            - float(control_preference["reward_margin_mean"]),
        },
        "interpretation": interpretation,
        "single_training_seed_feasibility_only": True,
        "length_ratios": {
            "synthetic": synthetic_length_ratio,
        },
    }


def run_stage4_confirm(
    config: dict[str, Any],
    *,
    search_run: str | Path,
    run_dir: str | Path | None = None,
) -> Path:
    """Perform the one-time confirmation after all models and files are locked."""

    search = Path(search_run).resolve()
    if not (search / "SEARCH_SUCCESS").is_file() or not (
        search / "SELECTION_LOCKED"
    ).is_file():
        raise ValueError("Stage 4 confirmation requires a completed locked search run")
    lock_path = search / "selection_lock.json"
    lock = read_json(lock_path)
    lock_hash = sha256_file(lock_path)
    search_manifest = read_json(search / "run_manifest.json")
    if (
        search_manifest.get("phase") != "stage4_search"
        or search_manifest.get("status") != "completed"
    ):
        raise ValueError("Stage 4 search manifest is not completed")
    if str(search_manifest.get("config_hash")) != config["_config_hash"]:
        raise ValueError("Stage 4 search manifest config hash mismatch")
    if str(search_manifest.get("selection_lock_sha256")) != lock_hash:
        raise ValueError("Stage 4 selection lock differs from the search manifest anchor")
    if (search / "SELECTION_LOCKED").read_text(encoding="utf-8").strip() != lock_hash:
        raise ValueError("Stage 4 selection lock differs from the immutable marker anchor")
    if lock.get("config_hash") != config["_config_hash"]:
        raise ValueError("Stage 4 confirmation config differs from the locked search config")
    verify_artifact_hashes(search, lock["search_evidence_sha256"])
    selection_result = read_json(search / "selection_result.json")
    control_selection = read_json(search / "control_selection.json")
    if str(selection_result.get("selected_candidate")) != str(
        lock["selected_candidate"]["id"]
    ):
        raise ValueError("Locked DPO candidate differs from the search selection result")
    if str(control_selection.get("selected_control")) != str(
        lock["selected_control"]["id"]
    ):
        raise ValueError("Locked SFT control differs from the search selection result")
    data_manifest = read_json(search / "frozen_data" / "stage4_data_manifest.json")
    if sha256_file(search / "frozen_data" / "stage4_data_manifest.json") != str(
        lock["data_manifest_sha256"]
    ):
        raise ValueError("Stage 4 data manifest differs from the locked search")
    expected_confirmation_roles = {
        "stage4_synthetic_confirmation",
        "stage4_preference_confirmation",
    }
    if set(lock.get("confirmation_files", {})) != expected_confirmation_roles:
        raise ValueError("Stage 4 lock has an unexpected confirmation role set")
    for role, item in lock["confirmation_files"].items():
        if str(item.get("source_dataset")) == "arkilpatel/SVAMP":
            raise ValueError("SVAMP is development-only and cannot enter confirmation")
        if data_manifest["roles"].get(role) != item:
            raise ValueError(f"Locked confirmation metadata differs for {role}")
    if sha256_file(search / "source_attestation.json") != str(
        lock["source_attestation_sha256"]
    ):
        raise ValueError("Stage 4 source attestation changed after selection")
    if sha256_file(search / "predecessor_attestation.json") != str(
        lock["predecessor_attestation_sha256"]
    ):
        raise ValueError("Stage 4 predecessor attestation changed after selection")
    implementation_path = search / "implementation_attestation.json"
    if sha256_file(implementation_path) != str(
        lock["implementation_attestation_sha256"]
    ):
        raise ValueError("Stage 4 implementation attestation changed after selection")
    search_environment_path = search / "environment.json"
    if sha256_file(search_environment_path) != str(lock["search_environment_sha256"]):
        raise ValueError("Stage 4 search environment snapshot changed after selection")
    locked_environment = read_json(search_environment_path)["environment"]
    locked_implementation = read_json(implementation_path)
    current_implementation = attest_stage4_implementation(config)
    if current_implementation["source_files"] != locked_implementation["source_files"]:
        raise ValueError("Stage 4 implementation source changed after selection")
    if current_implementation["combined_sha256"] != str(
        lock["implementation_combined_sha256"]
    ):
        raise ValueError("Stage 4 implementation digest differs from the locked search")
    fresh_attestation = attest_stage3_source(config)
    old_attestation = read_json(search / "source_attestation.json")
    for key in (
        "source_stage3_selection_lock_sha256",
        "source_stage3_integrity_report_sha256",
        "parent_adapter_sha256",
        "base_model_revision",
    ):
        if fresh_attestation[key] != old_attestation[key]:
            raise ValueError(f"Stage 3 source attestation changed: {key}")
    parent = Path(lock["parent"]["adapter"]).resolve()
    candidate_adapter = Path(lock["selected_candidate"]["adapter"]).resolve()
    control_adapter = Path(lock["selected_control"]["adapter"]).resolve()
    for name, path, expected in (
        ("parent", parent, lock["parent"]["adapter_sha256"]),
        ("DPO candidate", candidate_adapter, lock["selected_candidate"]["adapter_sha256"]),
        ("extra-SFT control", control_adapter, lock["selected_control"]["adapter_sha256"]),
    ):
        if sha256_directory(path) != str(expected):
            raise ValueError(f"Locked {name} adapter hash mismatch")
    for role, item in lock["confirmation_files"].items():
        verify_locked_file(item, expected_role=role)

    destination = _prepare_new_run_dir(run_dir or _new_run_dir(config, "confirm"))
    manifest_path = destination / "run_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "phase": "stage4_confirmation",
        "started_at": _utc_now(),
        "run_dir": str(destination),
        "search_run": str(search),
        "selection_lock_sha256": lock_hash,
        "stages": {},
    }
    write_json(manifest_path, manifest)
    _write_resolved_config(config, destination / "resolved_config.yaml")
    write_json(destination / "selection_lock.json", lock)
    try:
        confirmation_environment = _stage(
            manifest,
            manifest_path,
            "environment_preflight",
            environment_snapshot,
        )
        if confirmation_environment != locked_environment:
            raise RuntimeError(
                "Python, package, CUDA, GPU, or driver environment changed after selection"
            )
        confirmation_gpu = _stage(
            manifest,
            manifest_path,
            "gpu_preflight_before_final_claim",
            run_gpu_preflight,
        )
        write_json(
            destination / "environment.json",
            {
                "environment": confirmation_environment,
                "preflight": confirmation_gpu,
                "completed_before_final_claim": True,
            },
        )
        claim = _stage(
            manifest,
            manifest_path,
            "claim_final_rows",
            lambda: claim_final_rows(
                config,
                locked_files=lock["confirmation_files"],
                selection_lock_sha256=lock_hash,
                confirmation_run=destination,
            ),
        )
        write_json(destination / "final_exposure_claim.json", claim)
        marker_time = _utc_now()
        (destination / "FINAL_EVALUATION_STARTED").write_text(
            marker_time + "\n", encoding="utf-8"
        )
        manifest["final_evaluation_started_at"] = marker_time
        manifest["training_allowed_after_final_marker"] = False
        write_json(manifest_path, manifest)

        revision = str(lock["model_revision"])
        cache_dir = resolve_path(config, "cache_dir")
        files = lock["confirmation_files"]
        adapters = {
            "parent": parent,
            "control": control_adapter,
            "dpo": candidate_adapter,
        }
        metrics: dict[str, Any] = {}
        for branch, adapter in adapters.items():
            metrics[branch] = _stage(
                manifest,
                manifest_path,
                f"evaluate_{branch}_confirmation",
                lambda branch=branch, adapter=adapter: _evaluate_confirmation_branch(
                    config,
                    revision=revision,
                    cache_dir=cache_dir,
                    files=files,
                    adapter=adapter,
                    output_root=destination / "evaluation" / branch,
                ),
            )
        spec = lock["selected_candidate"]["spec"]
        metrics["dpo_preference"] = _stage(
            manifest,
            manifest_path,
            "score_dpo_preference_confirmation",
            lambda: evaluate_preference_adapter(
                config,
                revision=revision,
                cache_dir=cache_dir,
                data_path=files["stage4_preference_confirmation"]["path"],
                output_dir=destination / "evaluation" / "dpo" / "preferences",
                policy_adapter=candidate_adapter,
                reference_adapter=parent,
                beta=float(spec["beta"]),
                loss_type=str(spec["loss_type"]),
                evaluation_role="stage4_preference_confirmation",
            ),
        )
        metrics["control_preference"] = _stage(
            manifest,
            manifest_path,
            "score_control_preference_confirmation",
            lambda: evaluate_preference_adapter(
                config,
                revision=revision,
                cache_dir=cache_dir,
                data_path=files["stage4_preference_confirmation"]["path"],
                output_dir=destination / "evaluation" / "control" / "preferences",
                policy_adapter=control_adapter,
                reference_adapter=parent,
                beta=float(spec["beta"]),
                loss_type=str(spec["loss_type"]),
                evaluation_role="stage4_preference_confirmation",
            ),
        )
        write_json(destination / "confirmation_metrics.json", metrics)

        samples = int(config["evaluation"]["bootstrap_samples"])
        seed = int(config["evaluation"]["bootstrap_seed"])
        comparisons: dict[str, Any] = {}
        for comparison_name, left, right in (
            ("parent_to_control", "parent", "control"),
            ("parent_to_dpo", "parent", "dpo"),
            ("control_to_dpo", "control", "dpo"),
        ):
            comparisons[comparison_name] = {}
            for dataset, directory in (("synthetic", "synthetic"),):
                comparisons[comparison_name][dataset] = _compare_paths(
                    destination / "evaluation" / left / directory / "predictions.jsonl",
                    destination / "evaluation" / right / directory / "predictions.jsonl",
                    samples=samples,
                    seed=seed,
                )
        write_json(destination / "comparisons.json", comparisons)
        verdict = _confirmation_verdict(
            config, metrics=metrics, comparisons=comparisons
        )
        write_json(destination / "verdict.json", verdict)
        write_stage4_confirmation_report(
            destination / "report.md",
            verdict=verdict,
            metrics=metrics,
            comparisons=comparisons,
            lock=lock,
        )
        (destination / "SUCCESS").write_text("completed\n", encoding="utf-8")
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "verdict": verdict,
                "primary_artifacts": collect_artifact_hashes(
                    destination,
                    [
                        "selection_lock.json",
                        "final_exposure_claim.json",
                        "confirmation_metrics.json",
                        "comparisons.json",
                        "verdict.json",
                        "report.md",
                    ],
                ),
            }
        )
        write_json(manifest_path, manifest)
        completion = record_final_claim_completion(
            config,
            claim_identity=str(claim["claim_identity"]),
            selection_lock_sha256=lock_hash,
            confirmation_run=destination,
        )
        write_json(destination / "final_claim_completion.json", completion)
        manifest["primary_artifacts"]["final_claim_completion.json"] = sha256_file(
            destination / "final_claim_completion.json"
        )
        write_json(manifest_path, manifest)
        return destination
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(),
            }
        )
        write_json(manifest_path, manifest)
        (destination / "FAILED").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise
