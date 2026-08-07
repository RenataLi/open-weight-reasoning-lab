from __future__ import annotations

import copy
import json
import statistics
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import yaml

from reasoning_lab.config import public_config, resolve_path
from reasoning_lab.data.sft_mixture import materialize_training_view
from reasoning_lab.data.stage3_benchmarks import prepare_stage3_frozen_data
from reasoning_lab.evaluation.generation import evaluate_model
from reasoning_lab.evaluation.metrics import compare_predictions
from reasoning_lab.io_utils import read_json, read_jsonl, sha256_file, write_json
from reasoning_lab.reproducibility import environment_snapshot, run_gpu_preflight
from reasoning_lab.stage2_pipeline import _prepare_new_run_dir, _resolve_model_revision
from reasoning_lab.stage3_selection import canonical_hash, select_candidate
from reasoning_lab.training.continued_sft import sha256_directory, train_continued_sft
from reasoning_lab.training.sft import train_sft


RESEARCH_SOURCES = [
    {
        "title": "Qwen3 training guide: reasoning/non-reasoning mixture",
        "url": "https://qwen.readthedocs.io/en/latest/training/unsloth.html",
    },
    {
        "title": "Qwen3 official blog: thinking-mode fusion",
        "url": "https://qwenlm.github.io/blog/qwen3/",
    },
    {
        "title": "Qwen3 Technical Report",
        "url": "https://arxiv.org/abs/2505.09388",
    },
    {
        "title": "TRL SFTTrainer documentation",
        "url": "https://huggingface.co/docs/trl/en/sft_trainer",
    },
    {
        "title": "Dynamic Fine-Tuning (DFT)",
        "url": "https://arxiv.org/abs/2508.05629",
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


def _synthetic_eval_config(config: dict[str, Any], count: int) -> dict[str, Any]:
    derived = copy.deepcopy(config)
    derived["evaluation"]["max_examples"] = count
    derived["evaluation"]["batch_size"] = int(
        config["stage3"].get("synthetic_batch_size", config["evaluation"]["batch_size"])
    )
    return derived


def _external_eval_config(config: dict[str, Any], count: int) -> dict[str, Any]:
    derived = copy.deepcopy(config)
    source = config["external_evaluation"]
    derived["evaluation"] = {
        "split": "stage3_external",
        "max_examples": count,
        "batch_size": int(config["stage3"].get("external_batch_size", source["batch_size"])),
        "max_prompt_tokens": int(source["max_prompt_tokens"]),
        "max_new_tokens": int(source["max_new_tokens"]),
        "load_in_4bit": bool(source.get("load_in_4bit", False)),
        "stop_strings": list(source.get("stop_strings", [])),
        "bootstrap_samples": int(source["bootstrap_samples"]),
        "bootstrap_seed": int(source["bootstrap_seed"]),
        "max_hit_max_new_tokens_rate": float(source["max_hit_max_new_tokens_rate"]),
    }
    return derived


def _candidate_config(
    config: dict[str, Any], candidate: dict[str, Any], seed: int
) -> dict[str, Any]:
    derived = copy.deepcopy(config)
    training = derived["training"]
    training.update(
        {
            "seed": int(seed),
            "data_seed": int(seed),
            "max_steps": int(candidate["max_steps"]),
            "learning_rate": float(candidate["learning_rate"]),
            "loss_type": str(candidate["loss_type"]),
            "completion_only_loss": True,
            "save_strategy": "no",
            "save_steps": int(candidate["max_steps"]),
            "save_total_limit": 1,
        }
    )
    return derived


def _source_stage2_run(config: dict[str, Any]) -> Path:
    raw = Path(str(config["stage3"]["source_stage2_run"]))
    source = raw if raw.is_absolute() else Path(config["_project_root"]) / raw
    source = source.resolve()
    manifest = read_json(source / "run_manifest.json")
    if manifest.get("status") != "completed" or not (source / "SUCCESS").is_file():
        raise ValueError("Stage 3 requires a completed immutable Stage 2 source run")
    expected_revision = str(config["model"]["revision"]).lower()
    if str(manifest["model"]["resolved_revision"]).lower() != expected_revision:
        raise ValueError("Stage 2 source model revision differs from Stage 3")
    return source


def _confirmation_identity(files: dict[str, Any]) -> dict[str, Any]:
    """Return the path-independent identity of the two held-out datasets."""

    identity: dict[str, Any] = {}
    for role in ("synthetic_confirmation", "external_confirmation"):
        if role not in files:
            raise KeyError(f"Missing confirmation file metadata for {role}")
        item = files[role]
        if str(item.get("role")) != role:
            raise ValueError(f"Confirmation role mismatch for {role}")
        identity[role] = {
            "count": int(item["count"]),
            "row_id_set_sha256": str(item["row_id_set_sha256"]),
            "sha256": str(item["sha256"]),
        }
    return identity


def _confirmation_exposure_key(files: dict[str, Any]) -> str:
    return canonical_hash(_confirmation_identity(files))


def _exposure_registry_root(config: dict[str, Any]) -> Path:
    return resolve_path(config, "runs_dir") / ".stage3_confirmation_exposures"


def _legacy_exposure(
    config: dict[str, Any], *, exposure_key: str
) -> Path | None:
    """Recognize confirmation runs created before the registry was introduced."""

    runs_root = resolve_path(config, "runs_dir")
    if not runs_root.is_dir():
        return None
    for manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        try:
            manifest = read_json(manifest_path)
            if (
                manifest.get("phase") != "confirmation"
                or not manifest.get("test_evaluation_started_at")
            ):
                continue
            lock_path = manifest_path.parent / "selection_lock.json"
            if not lock_path.is_file():
                continue
            lock = read_json(lock_path)
            if _confirmation_exposure_key(lock["confirmation_files"]) == exposure_key:
                return manifest_path.parent
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return None


def _assert_confirmation_unexposed(
    config: dict[str, Any], files: dict[str, Any]
) -> str:
    """Refuse reuse of confirmation rows after their first evaluation exposure."""

    exposure_key = _confirmation_exposure_key(files)
    record_path = _exposure_registry_root(config) / f"{exposure_key}.json"
    if record_path.exists():
        raise FileExistsError(
            f"Confirmation datasets were already claimed for evaluation: {record_path}"
        )
    legacy = _legacy_exposure(config, exposure_key=exposure_key)
    if legacy is not None:
        raise FileExistsError(
            f"Confirmation datasets were already exposed by an earlier run: {legacy}"
        )
    return exposure_key


def _claim_confirmation_exposure(
    config: dict[str, Any],
    *,
    lock: dict[str, Any],
    lock_hash: str,
    search_run: Path,
    confirmation_run: Path,
) -> dict[str, Any]:
    """Atomically claim a held-out pair exactly once before evaluation starts."""

    files = lock["confirmation_files"]
    exposure_key = _assert_confirmation_unexposed(config, files)
    registry_root = _exposure_registry_root(config)
    registry_root.mkdir(parents=True, exist_ok=True)
    record_path = registry_root / f"{exposure_key}.json"
    record = {
        "claimed_at": _utc_now(),
        "confirmation_identity": _confirmation_identity(files),
        "confirmation_run": str(confirmation_run.resolve()),
        "exposure_key": exposure_key,
        "search_run": str(search_run.resolve()),
        "selection_lock_sha256": lock_hash,
    }
    try:
        with record_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"Confirmation datasets were claimed concurrently: {record_path}"
        ) from error
    return {**record, "registry_path": str(record_path.resolve())}


def _relative_file_hashes(root: Path, files: set[Path]) -> dict[str, str]:
    resolved_root = root.resolve()
    hashes: dict[str, str] = {}
    for path in sorted(files, key=lambda item: item.as_posix()):
        resolved = path.resolve()
        if resolved_root not in resolved.parents:
            raise ValueError(f"Evidence path escapes its run directory: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing evidence file: {resolved}")
        hashes[resolved.relative_to(resolved_root).as_posix()] = sha256_file(resolved)
    return hashes


def _search_evidence_hashes(destination: Path) -> dict[str, str]:
    """Hash every file used to train, score, and select a search candidate."""

    files = {
        destination / "frozen_data" / "frozen_data_manifest.json",
        destination / "frozen_data" / "synthetic_validation.jsonl",
        destination / "frozen_data" / "gsm8k_train_validation.jsonl",
        destination / "mixture_manifests.json",
        destination / "training_summaries.json",
        destination / "search_protocol.json",
        destination / "selection_result.json",
    }
    for root in (destination / "evaluation", destination / "mixtures"):
        if root.exists():
            files.update(path for path in root.rglob("*") if path.is_file())
    candidates_root = destination / "candidates"
    if candidates_root.exists():
        files.update(candidates_root.rglob("training_summary.json"))
    return _relative_file_hashes(destination, files)


def _verify_search_evidence(search: Path, lock: dict[str, Any]) -> None:
    expected = lock.get("search_evidence_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Selection lock has no complete search-evidence hash map")
    resolved_search = search.resolve()
    for relative, expected_hash in expected.items():
        path = (resolved_search / str(relative)).resolve()
        if resolved_search not in path.parents or not path.is_file():
            raise ValueError(f"Locked search evidence path is invalid: {relative}")
        if sha256_file(path) != str(expected_hash):
            raise ValueError(f"Locked search evidence changed: {relative}")


def _confirmation_primary_artifact_hashes(destination: Path) -> dict[str, str]:
    files = {
        destination / "environment.json",
        destination / "resolved_config.yaml",
        destination / "selection_lock.json",
        destination / "seed_results.json",
        destination / "comparisons.json",
        destination / "readiness.json",
        destination / "report.md",
    }
    for root in (destination / "evaluation", destination / "seeds"):
        if root.exists():
            files.update(
                path
                for path in root.rglob("*")
                if path.is_file()
                and (
                    path.name in {"metrics.json", "predictions.jsonl", "training_summary.json"}
                )
            )
    hashes = _relative_file_hashes(destination, files)
    for adapter in sorted(destination.glob("seeds/seed_*/continued_sft/final_adapter")):
        relative = adapter.resolve().relative_to(destination.resolve()).as_posix()
        hashes[f"{relative}/.directory_sha256"] = sha256_directory(adapter)
    return hashes


def _write_search_report(
    path: Path,
    *,
    config: dict[str, Any],
    model_revision: str,
    selection: dict[str, Any],
    base_synthetic: dict[str, Any],
    base_external: dict[str, Any],
    frozen_manifest: dict[str, Any],
) -> None:
    loss_types = sorted(
        {str(candidate["loss_type"]).upper() for candidate in config["stage3"]["candidates"]}
    )
    step_budgets = sorted(
        {int(candidate["max_steps"]) for candidate in config["stage3"]["candidates"]}
    )
    objective_sentence = (
        f"The preregistered objective matrix includes {', '.join(loss_types)}."
        if len(loss_types) > 1
        else f"All candidates in this round use {loss_types[0]}; the search isolates mixture and prompt conditioning."
    )
    compute_sentence = (
        "All cells use the same optimizer-step budget."
        if len(step_budgets) == 1
        else f"The preregistered optimizer-step budgets are {step_budgets}."
    )
    lines = [
        "# Stage 3 — SFT mixture/objective search",
        "",
        "## Outcome",
        "",
        f"- Selection status: **{selection['status']}**",
        f"- Locked candidate: **{selection['selected_candidate'] or 'none'}**",
        f"- Base revision: `{model_revision}`",
        f"- Search seeds: `{config['stage3']['search_seeds']}`",
        f"- Primary synthetic contract: `{frozen_manifest['synthetic_contract']}`",
        "- Candidate choice used validation roles only; fresh confirmation labels were not used.",
        "",
        "## Frozen data",
        "",
        f"- Synthetic validation: {frozen_manifest['files']['synthetic_validation']['count']}",
        f"- Fresh synthetic confirmation (not Stage 2 test): {frozen_manifest['files']['synthetic_confirmation']['count']}",
        f"- GSM8K train-derived validation: {frozen_manifest['files']['external_train_validation']['count']}",
        f"- Fresh GSM8K confirmation, excluding Stage 2 test subset: {frozen_manifest['files']['external_confirmation']['count']}",
        "",
        "## Validation results",
        "",
        f"Base synthetic exact: {base_synthetic['overall']['answer_exact_match']:.1%}; "
        f"base GSM8K-dev exact: {base_external['overall']['answer_exact_match']:.1%}.",
        "",
        "| Candidate | Eligible | Worst retention | Mean synthetic strict | Mean GSM8K-dev exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in selection["records"]:
        ranking = record["ranking"]
        lines.append(
            f"| {record['id']} | {'yes' if record['eligible'] else 'no'} | "
            f"{ranking['worst_seed_retention']:+.1%} | "
            f"{ranking['mean_synthetic_strict']:.1%} | "
            f"{ranking['mean_external_exact']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage 2 compressed synthetic generations from long worked solutions to short direct "
            "answers and lost multi-step accuracy. This search therefore varies the share of "
            "programmatically verified concise-reasoning targets. "
            f"{objective_sentence} {compute_sentence} All cells keep the same examples, QLoRA "
            "shape, and seeds; each candidate records its prompt-conditioning rule explicitly.",
            "",
            "This is fixed-example and step-matched, not token/compute-matched: reasoning completions "
            "contain more supervised tokens.",
            "",
            "## Primary sources",
            "",
        ]
    )
    lines.extend(f"- [{item['title']}]({item['url']})" for item in RESEARCH_SOURCES)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage3_search(
    config: dict[str, Any], *, run_dir: str | Path | None = None
) -> Path:
    if "stage3" not in config:
        raise ValueError("Missing stage3 configuration")
    destination = _prepare_new_run_dir(run_dir or _new_run_dir(config, "search"))
    manifest: dict[str, Any] = {
        "status": "running",
        "phase": "search",
        "started_at": _utc_now(),
        "config_hash": config["_config_hash"],
        "run_dir": str(destination),
        "stages": {},
    }
    manifest_path = destination / "run_manifest.json"
    write_json(manifest_path, manifest)
    _write_resolved_config(config, destination / "resolved_config.yaml")
    try:
        write_json(
            destination / "environment.json",
            _stage(manifest, manifest_path, "environment", environment_snapshot),
        )
        manifest["preflight"] = _stage(
            manifest, manifest_path, "preflight", run_gpu_preflight
        )
        model_revision = _stage(
            manifest,
            manifest_path,
            "resolve_model_revision",
            lambda: _resolve_model_revision(config),
        )
        source_stage2 = _stage(
            manifest,
            manifest_path,
            "verify_stage2_source",
            lambda: _source_stage2_run(config),
        )
        manifest["model_revision"] = model_revision
        manifest["source_stage2_run"] = str(source_stage2)

        frozen_manifest = _stage(
            manifest,
            manifest_path,
            "freeze_stage3_data",
            lambda: prepare_stage3_frozen_data(
                config,
                destination / "frozen_data",
                source_stage2_run=source_stage2,
            ),
        )
        _stage(
            manifest,
            manifest_path,
            "verify_confirmation_not_previously_exposed",
            lambda: _assert_confirmation_unexposed(config, frozen_manifest["files"]),
        )
        stage3 = config["stage3"]
        policy = {**stage3["selection_policy"], "expected_seeds": stage3["search_seeds"]}
        protocol = {
            "protocol_id": str(stage3["protocol_id"]),
            "created_at": _utc_now(),
            "model_revision": model_revision,
            "candidates": stage3["candidates"],
            "seeds": stage3["search_seeds"],
            "selection_policy": policy,
            "selection_roles": ["synthetic_validation", "external_train_validation"],
            "validation_files_precommitted": {
                role: frozen_manifest["files"][role]
                for role in ("synthetic_validation", "external_train_validation")
            },
            "confirmation_files_precommitted": {
                role: frozen_manifest["files"][role]
                for role in ("synthetic_confirmation", "external_confirmation")
            },
            "research_sources": RESEARCH_SOURCES,
        }
        write_json(destination / "search_protocol.json", protocol)

        original_data = resolve_path(config, "data_dir")
        mixture_manifests: dict[str, Any] = {}
        train_paths: dict[str, Path] = {}
        for candidate in stage3["candidates"]:
            candidate_id = str(candidate["id"])
            train_path = destination / "mixtures" / candidate_id / "train.jsonl"
            mixture_manifests[candidate_id] = _stage(
                manifest,
                manifest_path,
                f"materialize_{candidate_id}",
                lambda candidate=candidate, train_path=train_path: materialize_training_view(
                    original_data / "train.jsonl",
                    train_path,
                    candidate_id=str(candidate["id"]),
                    reasoning_fraction=float(candidate["reasoning_fraction"]),
                    mixture_seed=int(stage3["mixture_seed"]),
                    prompt_conditioning=str(
                        candidate.get("prompt_conditioning", "shared_flexible")
                    ),
                ),
            )
            train_paths[candidate_id] = train_path
        write_json(destination / "mixture_manifests.json", mixture_manifests)

        synth_config = _synthetic_eval_config(config, int(stage3["synthetic_dev_size"]))
        external_config = _external_eval_config(config, int(stage3["external_dev_size"]))
        synth_dev = destination / "frozen_data" / "synthetic_validation.jsonl"
        external_dev = destination / "frozen_data" / "gsm8k_train_validation.jsonl"
        base_synthetic = _stage(
            manifest,
            manifest_path,
            "evaluate_base_synthetic_validation",
            lambda: evaluate_model(
                synth_config,
                revision=model_revision,
                cache_dir=resolve_path(config, "cache_dir"),
                data_path=synth_dev,
                output_dir=destination / "evaluation" / "base" / "synthetic_validation",
                evaluation_role="synthetic_validation",
            ),
        )
        base_external = _stage(
            manifest,
            manifest_path,
            "evaluate_base_external_validation",
            lambda: evaluate_model(
                external_config,
                revision=model_revision,
                cache_dir=resolve_path(config, "cache_dir"),
                data_path=external_dev,
                output_dir=destination / "evaluation" / "base" / "external_validation",
                evaluation_role="external_train_validation",
            ),
        )

        candidate_records: list[dict[str, Any]] = []
        training_summaries: dict[str, Any] = {}
        for candidate in stage3["candidates"]:
            candidate_id = str(candidate["id"])
            seed_records: list[dict[str, Any]] = []
            training_summaries[candidate_id] = {}
            for seed_value in stage3["search_seeds"]:
                seed = int(seed_value)
                run_config = _candidate_config(config, candidate, seed)
                train_root = destination / "candidates" / candidate_id / f"seed_{seed}" / "sft"
                summary = _stage(
                    manifest,
                    manifest_path,
                    f"train_{candidate_id}_seed_{seed}",
                    lambda run_config=run_config, train_root=train_root: train_sft(
                        run_config,
                        revision=model_revision,
                        cache_dir=resolve_path(config, "cache_dir"),
                        train_path=train_paths[candidate_id],
                        output_dir=train_root,
                        resume=False,
                        require_fresh=True,
                    ),
                )
                training_summaries[candidate_id][str(seed)] = summary
                adapter = train_root / "final_adapter"
                synthetic = _stage(
                    manifest,
                    manifest_path,
                    f"eval_synth_{candidate_id}_seed_{seed}",
                    lambda adapter=adapter: evaluate_model(
                        synth_config,
                        revision=model_revision,
                        cache_dir=resolve_path(config, "cache_dir"),
                        data_path=synth_dev,
                        output_dir=destination
                        / "evaluation"
                        / candidate_id
                        / f"seed_{seed}"
                        / "synthetic_validation",
                        adapter_path=adapter,
                        evaluation_role="synthetic_validation",
                    ),
                )
                external = _stage(
                    manifest,
                    manifest_path,
                    f"eval_external_{candidate_id}_seed_{seed}",
                    lambda adapter=adapter: evaluate_model(
                        external_config,
                        revision=model_revision,
                        cache_dir=resolve_path(config, "cache_dir"),
                        data_path=external_dev,
                        output_dir=destination
                        / "evaluation"
                        / candidate_id
                        / f"seed_{seed}"
                        / "external_validation",
                        adapter_path=adapter,
                        evaluation_role="external_train_validation",
                    ),
                )
                seed_records.append(
                    {"seed": seed, "synthetic": synthetic, "external": external}
                )
            candidate_records.append({"id": candidate_id, "seeds": seed_records})
        write_json(destination / "training_summaries.json", training_summaries)
        selection = _stage(
            manifest,
            manifest_path,
            "validation_only_selection",
            lambda: select_candidate(
                base_synthetic=base_synthetic,
                base_external=base_external,
                candidates=candidate_records,
                policy=policy,
            ),
        )
        write_json(destination / "selection_result.json", selection)

        selected_id = selection["selected_candidate"]
        if selected_id is not None:
            selected_spec = next(
                item for item in stage3["candidates"] if item["id"] == selected_id
            )
            continuation_path = destination / "mixtures" / selected_id / "continuation.jsonl"
            continuation_manifest = _stage(
                manifest,
                manifest_path,
                "materialize_locked_continuation",
                lambda: materialize_training_view(
                    original_data / "continuation.jsonl",
                    continuation_path,
                    candidate_id=selected_id,
                    reasoning_fraction=float(selected_spec["reasoning_fraction"]),
                    mixture_seed=int(stage3["mixture_seed"]),
                    prompt_conditioning=str(
                        selected_spec.get("prompt_conditioning", "shared_flexible")
                    ),
                ),
            )
            selected_adapters = []
            for seed_value in stage3["search_seeds"]:
                seed = int(seed_value)
                summary = training_summaries[selected_id][str(seed)]
                selected_adapters.append(
                    {
                        "seed": seed,
                        "path": summary["final_adapter"],
                        "sha256": summary["final_adapter_sha256"],
                        "training_summary_sha256": sha256_file(
                            destination
                            / "candidates"
                            / selected_id
                            / f"seed_{seed}"
                            / "sft"
                            / "training_summary.json"
                        ),
                    }
                )
            lock = {
                "selection_locked_at": _utc_now(),
                "config_hash": config["_config_hash"],
                "model_revision": model_revision,
                "search_protocol_sha256": sha256_file(
                    destination / "search_protocol.json"
                ),
                "selection_result_sha256": sha256_file(
                    destination / "selection_result.json"
                ),
                "selected_candidate": selected_id,
                "selected_candidate_spec": selected_spec,
                "selected_adapters": selected_adapters,
                "locked_continuation": continuation_manifest["output"],
                "validation_files": {
                    role: frozen_manifest["files"][role]
                    for role in ("synthetic_validation", "external_train_validation")
                },
                "confirmation_files": {
                    role: frozen_manifest["files"][role]
                    for role in ("synthetic_confirmation", "external_confirmation")
                },
                "selection_input_roles": selection["selection_input_roles"],
                "test_metrics_used_for_selection": False,
            }
            lock["search_evidence_sha256"] = _search_evidence_hashes(destination)
            write_json(destination / "selection_lock.json", lock)
            lock_hash = sha256_file(destination / "selection_lock.json")
            (destination / "SELECTION_LOCKED").write_text(
                lock_hash + "\n", encoding="utf-8"
            )
            manifest["selection_lock_sha256"] = lock_hash
        _write_search_report(
            destination / "report.md",
            config=config,
            model_revision=model_revision,
            selection=selection,
            base_synthetic=base_synthetic,
            base_external=base_external,
            frozen_manifest=frozen_manifest,
        )
        manifest["selection"] = selection
        search_primary_files = {
            destination / "environment.json",
            destination / "resolved_config.yaml",
            destination / "report.md",
            destination / "selection_result.json",
        }
        if selected_id is not None:
            search_primary_files.add(destination / "selection_lock.json")
        manifest["primary_artifacts"] = {
            **(
                lock["search_evidence_sha256"]
                if selected_id is not None
                else _search_evidence_hashes(destination)
            ),
            **_relative_file_hashes(destination, search_primary_files),
        }
        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        write_json(manifest_path, manifest)
        (destination / "SEARCH_SUCCESS").write_text("completed\n", encoding="utf-8")
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


def _compare_files(
    baseline_path: Path,
    candidate_path: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    return compare_predictions(
        read_jsonl(baseline_path),
        read_jsonl(candidate_path),
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )


def _assert_metrics_provenance(
    metrics: dict[str, Any],
    *,
    locked_file: dict[str, Any],
    expected_role: str,
    model_revision: str,
) -> None:
    evaluation_input = metrics.get("evaluation_input", {})
    expected = {
        "role": expected_role,
        "count": int(locked_file["count"]),
        "sha256": str(locked_file["sha256"]),
        "row_id_set_sha256": str(locked_file["row_id_set_sha256"]),
    }
    observed = {
        "role": str(evaluation_input.get("role", "")),
        "count": int(evaluation_input.get("count", -1)),
        "sha256": str(evaluation_input.get("sha256", "")),
        "row_id_set_sha256": str(evaluation_input.get("row_id_set_sha256", "")),
    }
    if observed != expected:
        raise ValueError(
            f"Confirmation metrics provenance mismatch for {expected_role}: {observed}"
        )
    if str(metrics.get("model_revision", "")).lower() != model_revision.lower():
        raise ValueError(f"Confirmation metrics model revision mismatch for {expected_role}")


def _continued_lineage_checks(
    config: dict[str, Any],
    *,
    lock: dict[str, Any],
    locked_adapter: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, bool]:
    summary = result["continued_training"]
    seed = int(result["seed"])
    final_adapter = Path(str(summary.get("final_adapter", "")))
    try:
        final_hash_matches = (
            sha256_directory(final_adapter) == str(summary.get("final_adapter_sha256", ""))
        )
    except (FileNotFoundError, ValueError):
        final_hash_matches = False
    return {
        "parent_adapter_hash": str(summary.get("parent_adapter_sha256", ""))
        == str(locked_adapter["sha256"]),
        "parent_adapter_path": Path(str(summary.get("parent_adapter", ""))).resolve()
        == Path(str(locked_adapter["path"])).resolve(),
        "base_model_id": str(summary.get("parent_base_model_id", ""))
        == str(config["model"]["id"]),
        "base_revision": str(summary.get("parent_base_model_revision", "")).lower()
        == str(lock["model_revision"]).lower(),
        "continuation_hash": str(summary.get("continuation_dataset_sha256", ""))
        == str(lock["locked_continuation"]["sha256"]),
        "fresh_optimizer": bool(summary.get("fresh_optimizer")),
        "fresh_scheduler": bool(summary.get("fresh_scheduler")),
        "fresh_global_step": int(summary.get("initial_global_step", -1)) == 0,
        "never_resumed": summary.get("resumed_from_checkpoint") is None,
        "configured_steps": int(summary.get("global_step", -1))
        == int(config["continuation"]["max_steps"]),
        "derived_seed": int(
            summary.get("resolved_training_settings", {}).get("seed", -1)
        )
        == seed + 10_000,
        "derived_data_seed": int(
            summary.get("resolved_training_settings", {}).get("data_seed", -1)
        )
        == seed + 10_000,
        "final_adapter_hash": final_hash_matches,
    }


def _confirmation_gate(
    config: dict[str, Any],
    *,
    lock: dict[str, Any],
    marker_time: str,
    base: dict[str, dict[str, Any]],
    seeds: list[dict[str, Any]],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    policy = config["stage3"]["confirmation_policy"]
    locked_at = datetime.fromisoformat(lock["selection_locked_at"])
    test_started = datetime.fromisoformat(marker_time)
    checks: dict[str, dict[str, Any]] = {}
    locked_adapters = lock["selected_adapters"]
    locked_seed_values = [int(item["seed"]) for item in locked_adapters]
    observed_seed_values = [int(item["seed"]) for item in seeds]
    if len(locked_seed_values) != len(set(locked_seed_values)):
        raise ValueError("Selection lock contains duplicate adapter seeds")
    if len(observed_seed_values) != len(set(observed_seed_values)):
        raise ValueError("Confirmation results contain duplicate seeds")
    if sorted(observed_seed_values) != sorted(locked_seed_values):
        raise ValueError(
            f"Confirmation seed set {sorted(observed_seed_values)} does not match "
            f"the lock {sorted(locked_seed_values)}"
        )
    locked_by_seed = {int(item["seed"]): item for item in locked_adapters}
    synthetic_file = lock["confirmation_files"]["synthetic_confirmation"]
    external_file = lock["confirmation_files"]["external_confirmation"]
    _assert_metrics_provenance(
        base["synthetic"],
        locked_file=synthetic_file,
        expected_role="synthetic_confirmation",
        model_revision=str(lock["model_revision"]),
    )
    _assert_metrics_provenance(
        base["external"],
        locked_file=external_file,
        expected_role="external_confirmation",
        model_revision=str(lock["model_revision"]),
    )
    checks["Selection lock predates test evaluation"] = {
        "passed": locked_at < test_started and not lock["test_metrics_used_for_selection"],
        "detail": f"locked {locked_at.isoformat()}, test started {test_started.isoformat()}",
    }
    checks["All locked seeds completed"] = {
        "passed": len(seeds) == len(locked_adapters),
        "detail": f"{len(seeds)} / {len(locked_adapters)} seeds",
    }
    synth_deltas: list[float] = []
    external_deltas: list[float] = []
    synth_exact: list[float] = []
    external_exact: list[float] = []
    for result in seeds:
        seed = int(result["seed"])
        for branch in ("sft", "continued_sft"):
            _assert_metrics_provenance(
                result[branch]["synthetic"],
                locked_file=synthetic_file,
                expected_role="synthetic_confirmation",
                model_revision=str(lock["model_revision"]),
            )
            _assert_metrics_provenance(
                result[branch]["external"],
                locked_file=external_file,
                expected_role="external_confirmation",
                model_revision=str(lock["model_revision"]),
            )
        sft_synth = result["sft"]["synthetic"]["overall"]
        sft_external = result["sft"]["external"]["overall"]
        cont_synth = result["continued_sft"]["synthetic"]["overall"]
        cont_external = result["continued_sft"]["external"]["overall"]
        synth_delta = float(sft_synth["answer_exact_match"]) - float(
            base["synthetic"]["overall"]["answer_exact_match"]
        )
        external_delta = float(sft_external["answer_exact_match"]) - float(
            base["external"]["overall"]["answer_exact_match"]
        )
        synth_deltas.append(synth_delta)
        external_deltas.append(external_delta)
        synth_exact.append(float(sft_synth["answer_exact_match"]))
        external_exact.append(float(sft_external["answer_exact_match"]))
        seed_checks = {
            "synthetic_retention": synth_delta
            >= -float(policy["max_synthetic_regression"]),
            "external_retention": external_delta
            >= -float(policy["max_external_regression"]),
            "format": float(sft_synth["format_valid_rate"])
            >= float(policy["min_format_valid_rate"]),
            "parse": min(
                float(sft_synth["parse_success_rate"]),
                float(sft_external["parse_success_rate"]),
            )
            >= float(policy["min_parse_success_rate"]),
            "token_limits": (
                float(sft_synth["hit_max_new_tokens_rate"])
                <= float(policy["max_synthetic_token_limit_rate"])
                and float(sft_external["hit_max_new_tokens_rate"])
                <= float(policy["max_external_token_limit_rate"])
            ),
            "continued_synthetic_retention": float(cont_synth["answer_exact_match"])
            - float(sft_synth["answer_exact_match"])
            >= -float(policy["max_continued_regression"]),
            "continued_external_retention": float(cont_external["answer_exact_match"])
            - float(sft_external["answer_exact_match"])
            >= -float(policy["max_continued_regression"]),
            "continued_format": float(cont_synth["format_valid_rate"])
            >= float(policy["min_format_valid_rate"]),
            "continued_parse": min(
                float(cont_synth["parse_success_rate"]),
                float(cont_external["parse_success_rate"]),
            )
            >= float(policy["min_parse_success_rate"]),
            "continued_token_limits": (
                float(cont_synth["hit_max_new_tokens_rate"])
                <= float(policy["max_synthetic_token_limit_rate"])
                and float(cont_external["hit_max_new_tokens_rate"])
                <= float(policy["max_external_token_limit_rate"])
            ),
        }
        lineage_checks = _continued_lineage_checks(
            config,
            lock=lock,
            locked_adapter=locked_by_seed[seed],
            result=result,
        )
        seed_checks["continued_lineage"] = all(lineage_checks.values())
        checks[f"Seed {seed} passes capability and format thresholds"] = {
            "passed": all(seed_checks.values()),
            "detail": {**seed_checks, "continued_lineage_detail": lineage_checks},
        }
        synth_ci = comparisons[f"seed_{seed}"]["base_to_sft_synthetic"][
            "paired_bootstrap_95_ci"
        ]
        external_ci = comparisons[f"seed_{seed}"]["base_to_sft_external"][
            "paired_bootstrap_95_ci"
        ]
        checks[f"Seed {seed} paired uncertainty clears guardrails"] = {
            "passed": (
                float(synth_ci[0]) > -float(policy["max_synthetic_ci_regression"])
                and float(external_ci[0])
                > -float(policy["max_external_ci_regression"])
            ),
            "detail": {"synthetic_95_ci": synth_ci, "external_95_ci": external_ci},
        }
    synth_range = max(synth_exact) - min(synth_exact)
    external_range = max(external_exact) - min(external_exact)
    checks["Seed stability is acceptable"] = {
        "passed": (
            synth_range <= float(policy["max_synthetic_seed_range"])
            and external_range <= float(policy["max_external_seed_range"])
        ),
        "detail": {
            "synthetic_range": synth_range,
            "external_range": external_range,
        },
    }
    passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "status": "READY TO BUILD A DPO PILOT" if passed else "NOT READY FOR DPO",
        "passed": passed,
        "checks": checks,
        "summary": {
            "base_synthetic_exact": base["synthetic"]["overall"]["answer_exact_match"],
            "sft_synthetic_exact_mean": statistics.fmean(synth_exact),
            "sft_synthetic_delta_mean": statistics.fmean(synth_deltas),
            "base_external_exact": base["external"]["overall"]["answer_exact_match"],
            "sft_external_exact_mean": statistics.fmean(external_exact),
            "sft_external_delta_mean": statistics.fmean(external_deltas),
        },
    }


def _write_confirmation_report(
    path: Path,
    *,
    lock: dict[str, Any],
    base: dict[str, dict[str, Any]],
    seeds: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> None:
    synthetic_count = int(
        lock["confirmation_files"]["synthetic_confirmation"]["count"]
    )
    external_count = int(
        lock["confirmation_files"]["external_confirmation"]["count"]
    )
    lines = [
        "# Stage 3 — locked multi-seed confirmation",
        "",
        "## Outcome",
        "",
        f"- **{readiness['status']}**",
        f"- Locked candidate: `{lock['selected_candidate']}`",
        f"- Base revision: `{lock['model_revision']}`",
        f"- Confirmation used {synthetic_count} new synthetic examples and "
        f"{external_count} GSM8K test examples not seen in Stage 2.",
        "- No candidate, objective, learning rate, or seed was changed after the selection lock.",
        "",
        "## Held-out results",
        "",
        "| Model | Seed | Synthetic exact | Synthetic format valid | Synthetic strict-correct | GSM8K exact |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Base | — | {base['synthetic']['overall']['answer_exact_match']:.1%} | "
        f"{base['synthetic']['overall']['format_valid_rate']:.1%} | "
        f"{base['synthetic']['overall']['strict_correct_rate']:.1%} | "
        f"{base['external']['overall']['answer_exact_match']:.1%} |",
    ]
    for result in seeds:
        for label, key in (("SFT", "sft"), ("continued-SFT", "continued_sft")):
            synth = result[key]["synthetic"]["overall"]
            external = result[key]["external"]["overall"]
            lines.append(
                f"| {label} | {result['seed']} | {synth['answer_exact_match']:.1%} | "
                f"{synth['format_valid_rate']:.1%} | {synth['strict_correct_rate']:.1%} | "
                f"{external['answer_exact_match']:.1%} |"
            )
    summary = readiness["summary"]
    lines.extend(
        [
            "",
            "Across locked SFT seeds: synthetic exact "
            f"{summary['sft_synthetic_exact_mean']:.1%} "
            f"({summary['sft_synthetic_delta_mean']:+.1%} vs Base); GSM8K exact "
            f"{summary['sft_external_exact_mean']:.1%} "
            f"({summary['sft_external_delta_mean']:+.1%} vs Base).",
            "",
            "## Readiness checks",
            "",
        ]
    )
    for name, check in readiness["checks"].items():
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'} — {name}: {check['detail']}")
    lines.extend(
        [
            "",
            "Passing this gate means only that the SFT foundation is stable enough to design a DPO "
            "pilot. It is not evidence that DPO itself improves reasoning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage3_confirm(
    config: dict[str, Any],
    *,
    search_run: str | Path,
    run_dir: str | Path | None = None,
) -> Path:
    search = Path(search_run).resolve()
    search_manifest = read_json(search / "run_manifest.json")
    if search_manifest.get("status") != "completed" or not (
        search / "SEARCH_SUCCESS"
    ).is_file():
        raise ValueError("Stage 3 confirmation requires a completed search run")
    if search_manifest.get("config_hash") != config["_config_hash"]:
        raise ValueError("Stage 3 search and confirmation config hashes differ")
    lock_path = search / "selection_lock.json"
    if not lock_path.is_file() or not (search / "SELECTION_LOCKED").is_file():
        raise ValueError("Stage 3 search did not produce an immutable selection lock")
    lock_hash = sha256_file(lock_path)
    marker_hash = (search / "SELECTION_LOCKED").read_text(encoding="utf-8").strip()
    if marker_hash != lock_hash or search_manifest.get("selection_lock_sha256") != lock_hash:
        raise ValueError("Stage 3 selection lock hash mismatch")
    lock = read_json(lock_path)
    if lock.get("selected_candidate") is None:
        raise ValueError("Stage 3 has no eligible locked candidate to confirm")
    if lock["config_hash"] != config["_config_hash"]:
        raise ValueError("Selection lock config hash mismatch")
    if sorted(lock.get("selection_input_roles", [])) != [
        "external_train_validation",
        "synthetic_validation",
    ] or bool(lock.get("test_metrics_used_for_selection", True)):
        raise ValueError("Selection lock does not prove validation-only selection")
    _verify_search_evidence(search, lock)
    if sha256_file(search / "search_protocol.json") != lock["search_protocol_sha256"]:
        raise ValueError("Locked search protocol changed")
    if sha256_file(search / "selection_result.json") != lock["selection_result_sha256"]:
        raise ValueError("Locked selection result changed")
    validation_files = lock.get("validation_files", {})
    if set(validation_files) != {
        "synthetic_validation",
        "external_train_validation",
    }:
        raise ValueError("Selection lock has an incomplete validation-data precommitment")
    for role, item in validation_files.items():
        if role not in {"synthetic_validation", "external_train_validation"}:
            raise ValueError(f"Unexpected locked validation role: {role}")
        if str(item.get("role")) != role or sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"Locked validation data changed for {role}")
    for role, item in lock["confirmation_files"].items():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Locked confirmation data changed for {role}")
    if sha256_file(lock["locked_continuation"]["path"]) != lock["locked_continuation"][
        "sha256"
    ]:
        raise ValueError("Locked continuation data changed")
    for adapter in lock["selected_adapters"]:
        if sha256_directory(adapter["path"]) != adapter["sha256"]:
            raise ValueError(f"Locked adapter changed for seed {adapter['seed']}")
        summary_path = Path(adapter["path"]).parent / "training_summary.json"
        if sha256_file(summary_path) != adapter["training_summary_sha256"]:
            raise ValueError(f"Locked training summary changed for seed {adapter['seed']}")

    destination = _prepare_new_run_dir(run_dir or _new_run_dir(config, "confirm"))
    manifest: dict[str, Any] = {
        "status": "running",
        "phase": "confirmation",
        "started_at": _utc_now(),
        "config_hash": config["_config_hash"],
        "search_run": str(search),
        "selection_lock_sha256": lock_hash,
        "run_dir": str(destination),
        "stages": {},
    }
    manifest_path = destination / "run_manifest.json"
    write_json(manifest_path, manifest)
    _write_resolved_config(config, destination / "resolved_config.yaml")
    write_json(destination / "selection_lock.json", lock)
    write_json(
        destination / "environment.json",
        _stage(manifest, manifest_path, "environment", environment_snapshot),
    )
    manifest["preflight"] = _stage(
        manifest, manifest_path, "preflight", run_gpu_preflight
    )
    try:
        exposure_claim = _claim_confirmation_exposure(
            config,
            lock=lock,
            lock_hash=lock_hash,
            search_run=search,
            confirmation_run=destination,
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["finished_at"] = _utc_now()
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(manifest_path, manifest)
        raise
    manifest["exposure_claim"] = exposure_claim
    marker_time = _utc_now()
    (destination / "TEST_EVALUATION_STARTED").write_text(
        marker_time + "\n", encoding="utf-8"
    )
    manifest["test_evaluation_started_at"] = marker_time
    write_json(manifest_path, manifest)
    try:
        model_revision = str(lock["model_revision"])
        cache_dir = resolve_path(config, "cache_dir")
        stage3 = config["stage3"]
        synth_path = Path(lock["confirmation_files"]["synthetic_confirmation"]["path"])
        external_path = Path(lock["confirmation_files"]["external_confirmation"]["path"])
        synth_config = _synthetic_eval_config(
            config, int(stage3["synthetic_confirm_size"])
        )
        external_config = _external_eval_config(
            config, int(stage3["external_confirm_size"])
        )
        base = {
            "synthetic": _stage(
                manifest,
                manifest_path,
                "evaluate_base_synthetic_confirmation",
                lambda: evaluate_model(
                    synth_config,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    data_path=synth_path,
                    output_dir=destination / "evaluation" / "base" / "synthetic",
                    evaluation_role="synthetic_confirmation",
                ),
            ),
            "external": _stage(
                manifest,
                manifest_path,
                "evaluate_base_external_confirmation",
                lambda: evaluate_model(
                    external_config,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    data_path=external_path,
                    output_dir=destination / "evaluation" / "base" / "external",
                    evaluation_role="external_confirmation",
                ),
            ),
        }
        selected_spec = lock["selected_candidate_spec"]
        seed_results: list[dict[str, Any]] = []
        comparisons: dict[str, Any] = {}
        for adapter_info in lock["selected_adapters"]:
            seed = int(adapter_info["seed"])
            adapter_path = Path(adapter_info["path"])
            seed_root = destination / "seeds" / f"seed_{seed}"
            sft_synthetic = _stage(
                manifest,
                manifest_path,
                f"eval_sft_seed_{seed}_synthetic",
                lambda adapter_path=adapter_path: evaluate_model(
                    synth_config,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    data_path=synth_path,
                    output_dir=seed_root / "evaluation" / "sft" / "synthetic",
                    adapter_path=adapter_path,
                    evaluation_role="synthetic_confirmation",
                ),
            )
            sft_external = _stage(
                manifest,
                manifest_path,
                f"eval_sft_seed_{seed}_external",
                lambda adapter_path=adapter_path: evaluate_model(
                    external_config,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    data_path=external_path,
                    output_dir=seed_root / "evaluation" / "sft" / "external",
                    adapter_path=adapter_path,
                    evaluation_role="external_confirmation",
                ),
            )
            continued_config = copy.deepcopy(config)
            continued_config["continuation"].update(
                {
                    "seed": seed + 10_000,
                    "data_seed": seed + 10_000,
                    "loss_type": str(selected_spec["loss_type"]),
                    "save_strategy": "no",
                    "save_steps": int(config["continuation"]["max_steps"]),
                    "save_total_limit": 1,
                }
            )
            continued_summary = _stage(
                manifest,
                manifest_path,
                f"train_continued_seed_{seed}",
                lambda continued_config=continued_config, adapter_path=adapter_path: train_continued_sft(
                    continued_config,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    continuation_path=lock["locked_continuation"]["path"],
                    parent_adapter=adapter_path,
                    output_dir=seed_root / "continued_sft",
                ),
            )
            continued_adapter = Path(continued_summary["final_adapter"])
            continued_synthetic = _stage(
                manifest,
                manifest_path,
                f"eval_continued_seed_{seed}_synthetic",
                lambda continued_adapter=continued_adapter: evaluate_model(
                    synth_config,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    data_path=synth_path,
                    output_dir=seed_root
                    / "evaluation"
                    / "continued_sft"
                    / "synthetic",
                    adapter_path=continued_adapter,
                    evaluation_role="synthetic_confirmation",
                ),
            )
            continued_external = _stage(
                manifest,
                manifest_path,
                f"eval_continued_seed_{seed}_external",
                lambda continued_adapter=continued_adapter: evaluate_model(
                    external_config,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    data_path=external_path,
                    output_dir=seed_root
                    / "evaluation"
                    / "continued_sft"
                    / "external",
                    adapter_path=continued_adapter,
                    evaluation_role="external_confirmation",
                ),
            )
            seed_results.append(
                {
                    "seed": seed,
                    "sft": {"synthetic": sft_synthetic, "external": sft_external},
                    "continued_sft": {
                        "synthetic": continued_synthetic,
                        "external": continued_external,
                    },
                    "continued_training": continued_summary,
                }
            )
            samples_synth = int(config["evaluation"]["bootstrap_samples"])
            samples_external = int(config["external_evaluation"]["bootstrap_samples"])
            comparisons[f"seed_{seed}"] = {
                "base_to_sft_synthetic": _compare_files(
                    destination / "evaluation" / "base" / "synthetic" / "predictions.jsonl",
                    seed_root / "evaluation" / "sft" / "synthetic" / "predictions.jsonl",
                    samples=samples_synth,
                    seed=int(config["evaluation"]["bootstrap_seed"]) + seed,
                ),
                "base_to_sft_external": _compare_files(
                    destination / "evaluation" / "base" / "external" / "predictions.jsonl",
                    seed_root / "evaluation" / "sft" / "external" / "predictions.jsonl",
                    samples=samples_external,
                    seed=int(config["external_evaluation"]["bootstrap_seed"]) + seed,
                ),
                "sft_to_continued_synthetic": _compare_files(
                    seed_root / "evaluation" / "sft" / "synthetic" / "predictions.jsonl",
                    seed_root
                    / "evaluation"
                    / "continued_sft"
                    / "synthetic"
                    / "predictions.jsonl",
                    samples=samples_synth,
                    seed=int(config["evaluation"]["bootstrap_seed"]) + seed + 1,
                ),
                "sft_to_continued_external": _compare_files(
                    seed_root / "evaluation" / "sft" / "external" / "predictions.jsonl",
                    seed_root
                    / "evaluation"
                    / "continued_sft"
                    / "external"
                    / "predictions.jsonl",
                    samples=samples_external,
                    seed=int(config["external_evaluation"]["bootstrap_seed"]) + seed + 1,
                ),
            }
        write_json(destination / "seed_results.json", seed_results)
        write_json(destination / "comparisons.json", comparisons)
        readiness = _stage(
            manifest,
            manifest_path,
            "confirmation_gate",
            lambda: _confirmation_gate(
                config,
                lock=lock,
                marker_time=marker_time,
                base=base,
                seeds=seed_results,
                comparisons=comparisons,
            ),
        )
        write_json(destination / "readiness.json", readiness)
        _write_confirmation_report(
            destination / "report.md",
            lock=lock,
            base=base,
            seeds=seed_results,
            readiness=readiness,
        )
        manifest["readiness"] = readiness
        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        manifest["primary_artifacts"] = _confirmation_primary_artifact_hashes(
            destination
        )
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
