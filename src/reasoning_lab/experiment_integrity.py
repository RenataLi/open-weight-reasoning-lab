from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reasoning_lab.io_utils import read_json, read_jsonl, sha256_file, write_json
from reasoning_lab.training.continued_sft import sha256_directory


STAGE4_IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "requirements.lock.txt",
    "scripts/run-stage4.ps1",
    "src/reasoning_lab/__init__.py",
    "src/reasoning_lab/cli.py",
    "src/reasoning_lab/config.py",
    "src/reasoning_lab/io_utils.py",
    "src/reasoning_lab/reproducibility.py",
    "src/reasoning_lab/preference_tokenization.py",
    "src/reasoning_lab/data/stage4_reasoning.py",
    "src/reasoning_lab/data/stage4_retired_v1.py",
    "src/reasoning_lab/data/svamp.py",
    "src/reasoning_lab/evaluation/generation.py",
    "src/reasoning_lab/evaluation/metrics.py",
    "src/reasoning_lab/evaluation/parsing.py",
    "src/reasoning_lab/evaluation/preferences.py",
    "src/reasoning_lab/models/loading.py",
    "src/reasoning_lab/reporting/stage4.py",
    "src/reasoning_lab/stage2_pipeline.py",
    "src/reasoning_lab/stage4_selection.py",
    "src/reasoning_lab/stage4_pipeline.py",
    "src/reasoning_lab/training/continued_sft.py",
    "src/reasoning_lab/training/dpo.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def row_id_set_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted(str(row["id"]) for row in rows),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attest_stage4_implementation(config: dict[str, Any]) -> dict[str, Any]:
    """Hash every local source/config file that can affect the Stage 4 result."""

    project_root = Path(config["_project_root"]).resolve()
    config_path = Path(config["_config_path"]).resolve()
    try:
        config_relative = config_path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise ValueError("Stage 4 config must live inside the project root") from error
    package_root = project_root / "src" / "reasoning_lab"
    package_sources = {
        path.relative_to(project_root).as_posix()
        for path in package_root.rglob("*.py")
    }
    relative_paths = sorted(
        {*STAGE4_IMPLEMENTATION_FILES, *package_sources, config_relative}
    )
    source_files: dict[str, str] = {}
    for relative in relative_paths:
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise ValueError(f"Stage 4 implementation path escapes project: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 4 implementation file: {path}")
        source_files[relative] = sha256_file(path)
    payload = json.dumps(
        source_files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "project_root": str(project_root),
        "config_relative_path": config_relative,
        "source_files": source_files,
        "combined_sha256": hashlib.sha256(payload).hexdigest(),
    }


def collect_artifact_hashes(root: str | Path, paths: list[str | Path]) -> dict[str, str]:
    base = Path(root).resolve()
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        absolute = path if path.is_absolute() else base / path
        absolute = absolute.resolve()
        try:
            relative = absolute.relative_to(base).as_posix()
        except ValueError as error:
            raise ValueError(f"Artifact is outside the declared root: {absolute}") from error
        if not absolute.is_file():
            raise FileNotFoundError(f"Missing evidence artifact: {absolute}")
        result[relative] = sha256_file(absolute)
    return dict(sorted(result.items()))


def verify_artifact_hashes(root: str | Path, expected: dict[str, str]) -> None:
    base = Path(root).resolve()
    for relative, digest in expected.items():
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ValueError(f"Evidence path escapes its root: {relative}") from error
        if not path.is_file() or sha256_file(path) != str(digest):
            raise ValueError(f"Evidence artifact changed or disappeared: {relative}")


def attest_stage3_source(config: dict[str, Any]) -> dict[str, Any]:
    """Re-audit the historical Stage 3 selection without modifying it."""

    project_root = Path(config["_project_root"])
    stage4 = config["stage4"]
    search_raw = Path(str(stage4["source_stage3_search"]))
    search = search_raw if search_raw.is_absolute() else project_root / search_raw
    search = search.resolve()
    lock_path = search / "selection_lock.json"
    expected_lock_hash = str(stage4["source_stage3_lock_sha256"])
    if sha256_file(lock_path) != expected_lock_hash:
        raise ValueError("Historical Stage 3 selection lock hash mismatch")
    lock = read_json(lock_path)
    expected_revision = str(config["model"]["revision"]).lower()
    if str(lock.get("model_revision", "")).lower() != expected_revision:
        raise ValueError("Historical Stage 3 base revision mismatch")
    selected = str(lock.get("selected_candidate", ""))
    adapters = sorted(lock.get("selected_adapters", []), key=lambda row: int(row["seed"]))
    if not adapters:
        raise ValueError("Historical Stage 3 lock has no selected adapters")
    expected_seed = int(stage4["parent_seed"])
    if expected_seed != int(adapters[0]["seed"]):
        raise ValueError("Stage 4 parent is not the preregistered minimum Stage 3 seed")
    locked_parent = next(
        (row for row in adapters if int(row["seed"]) == expected_seed), None
    )
    if locked_parent is None:
        raise ValueError("Stage 4 parent seed is absent from the Stage 3 lock")
    parent_raw = Path(str(stage4["parent_adapter"]))
    parent = parent_raw if parent_raw.is_absolute() else project_root / parent_raw
    parent = parent.resolve()
    locked_path = Path(str(locked_parent["path"])).resolve()
    if parent != locked_path:
        raise ValueError("Configured Stage 4 parent path differs from the Stage 3 lock")
    parent_hash = sha256_directory(parent)
    if parent_hash != str(locked_parent["sha256"]) or parent_hash != str(
        stage4["parent_adapter_sha256"]
    ):
        raise ValueError("Stage 4 parent adapter hash mismatch")

    audit_raw = Path(str(stage4["source_stage3_integrity_report"]))
    audit_path = audit_raw if audit_raw.is_absolute() else project_root / audit_raw
    audit_path = audit_path.resolve()
    audit = read_json(audit_path)
    if audit.get("status") != "passed":
        raise ValueError("Historical Stage 3 integrity report did not pass")
    if str(audit.get("checks", {}).get("selection_lock_sha256")) != expected_lock_hash:
        raise ValueError("Stage 3 integrity audit references a different selection lock")

    artifact_locations = {
        "search_protocol.json": search / "search_protocol.json",
        "selection_result.json": search / "selection_result.json",
        "selection_lock.json": search / "selection_lock.json",
        "frozen_data_manifest.json": search / "frozen_data" / "frozen_data_manifest.json",
        "mixture_manifests.json": search / "mixture_manifests.json",
        "training_summaries.json": search / "training_summaries.json",
        "synthetic_validation.jsonl": search / "frozen_data" / "synthetic_validation.jsonl",
        "gsm8k_train_validation.jsonl": search / "frozen_data" / "gsm8k_train_validation.jsonl",
        "synthetic_confirmation.jsonl": search / "frozen_data" / "synthetic_confirmation.jsonl",
        "gsm8k_confirmation.jsonl": search / "frozen_data" / "gsm8k_confirmation.jsonl",
    }
    audited_artifacts = audit.get("search_artifacts", {})
    if set(artifact_locations) != set(audited_artifacts):
        raise ValueError("Stage 3 audit evidence inventory is incomplete or unexpected")
    for name, path in artifact_locations.items():
        if sha256_file(path) != str(audited_artifacts[name]):
            raise ValueError(f"Historical Stage 3 artifact changed: {name}")

    return {
        "attested_at": _utc_now(),
        "source_stage3_search": str(search),
        "source_stage3_selection_lock": str(lock_path),
        "source_stage3_selection_lock_sha256": expected_lock_hash,
        "source_stage3_integrity_report": str(audit_path),
        "source_stage3_integrity_report_sha256": sha256_file(audit_path),
        "selected_candidate": selected,
        "parent_selection_rule": str(stage4["parent_selection_rule"]),
        "parent_seed": expected_seed,
        "parent_adapter": str(parent),
        "parent_adapter_sha256": parent_hash,
        "base_model_revision": expected_revision,
        "verified_search_artifacts": {
            name: str(audited_artifacts[name]) for name in sorted(audited_artifacts)
        },
    }


def verify_locked_file(item: dict[str, Any], *, expected_role: str) -> list[dict[str, Any]]:
    if str(item.get("role")) != expected_role:
        raise ValueError(f"Locked role mismatch: expected {expected_role}")
    path = Path(str(item["path"])).resolve()
    if sha256_file(path) != str(item["sha256"]):
        raise ValueError(f"Locked file hash mismatch for {expected_role}")
    rows = read_jsonl(path)
    if len(rows) != int(item["count"]):
        raise ValueError(f"Locked row count mismatch for {expected_role}")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError(f"Locked file contains duplicate ids: {expected_role}")
    if row_id_set_sha256(rows) != str(item["row_id_set_sha256"]):
        raise ValueError(f"Locked id-set hash mismatch for {expected_role}")
    return rows


def _completed_confirmation_runs(
    runs_root: Path, *, selection_lock_sha256: str
) -> list[str]:
    completed: list[str] = []
    for manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            manifest.get("phase") == "stage4_confirmation"
            and manifest.get("status") == "completed"
            and str(manifest.get("selection_lock_sha256"))
            == selection_lock_sha256
            and (manifest_path.parent / "SUCCESS").is_file()
        ):
            completed.append(str(manifest_path.parent.resolve()))
    return completed


def _acquire_confirmation_lease(
    registry: Path,
    *,
    claim_identity: str,
    selection_lock_sha256: str,
    confirmation_run: Path,
) -> dict[str, Any]:
    """Atomically prevent two confirmation processes from evaluating one lock."""

    active_root = registry / "active"
    active_root.mkdir(parents=True, exist_ok=True)
    lease_path = active_root / f"{claim_identity}.json"
    record = {
        "acquired_at": _utc_now(),
        "claim_identity": claim_identity,
        "confirmation_run": str(confirmation_run),
        "selection_lock_sha256": selection_lock_sha256,
    }
    while True:
        try:
            with lease_path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            return record
        except FileExistsError as error:
            previous = read_json(lease_path)
            previous_run = Path(str(previous.get("confirmation_run", "")))
            if previous_run.resolve() == confirmation_run:
                return previous
            try:
                previous_manifest = read_json(previous_run / "run_manifest.json")
            except (OSError, ValueError, json.JSONDecodeError):
                previous_manifest = {}
            retryable = (
                previous_manifest.get("phase") == "stage4_confirmation"
                and previous_manifest.get("status") in {"failed", "aborted"}
            )
            if not retryable:
                raise FileExistsError(
                    "Another confirmation process already owns this selection lock: "
                    f"{previous_run or lease_path}"
                ) from error
            try:
                lease_path.unlink()
            except FileNotFoundError:
                pass


def claim_final_rows(
    config: dict[str, Any],
    *,
    locked_files: dict[str, dict[str, Any]],
    selection_lock_sha256: str,
    confirmation_run: str | Path,
) -> dict[str, Any]:
    """Atomically claim each final row; partial claims remain exposed by design."""

    runs_raw = Path(config["paths"]["runs_dir"])
    runs_root = runs_raw if runs_raw.is_absolute() else Path(config["_project_root"]) / runs_raw
    registry = runs_root.resolve() / ".stage4_final_exposures"
    row_root = registry / "rows"
    claim_root = registry / "claims"
    row_root.mkdir(parents=True, exist_ok=True)
    claim_root.mkdir(parents=True, exist_ok=True)

    completed_runs = _completed_confirmation_runs(
        runs_root.resolve(), selection_lock_sha256=selection_lock_sha256
    )
    if completed_runs:
        raise FileExistsError(
            "This selection lock already has a successful one-time confirmation: "
            f"{completed_runs[0]}"
        )

    materialized: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[tuple[str, str]] = []
    for role, item in sorted(locked_files.items()):
        rows = verify_locked_file(item, expected_role=role)
        materialized[role] = rows
        all_ids.extend((role, str(row["id"])) for row in rows)
    raw_ids = [item_id for _, item_id in all_ids]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("Locked final datasets overlap with each other")

    claim_identity = hashlib.sha256(
        json.dumps(
            {
                "ids": sorted(raw_ids),
                "selection_lock_sha256": selection_lock_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    completion_marker = registry / "completed" / f"{claim_identity}.json"
    if completion_marker.is_file():
        previous_completion = read_json(completion_marker)
        raise FileExistsError(
            "This final claim already has a successful one-time confirmation: "
            f"{previous_completion.get('confirmation_run', completion_marker)}"
        )
    confirmation_path = Path(confirmation_run).resolve()
    lease = _acquire_confirmation_lease(
        registry,
        claim_identity=claim_identity,
        selection_lock_sha256=selection_lock_sha256,
        confirmation_run=confirmation_path,
    )
    created_at = _utc_now()
    existing_records: list[dict[str, Any]] = []
    for role, item_id in sorted(all_ids, key=lambda value: value[1]):
        row_key = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
        marker = row_root / f"{row_key}.json"
        record = {
            "claimed_at": created_at,
            "claim_identity": claim_identity,
            "confirmation_run": str(Path(confirmation_run).resolve()),
            "id": item_id,
            "role": role,
            "selection_lock_sha256": selection_lock_sha256,
        }
        try:
            with marker.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError as error:
            previous = read_json(marker)
            same_claim = (
                str(previous.get("selection_lock_sha256"))
                == selection_lock_sha256
                and str(previous.get("claim_identity")) == claim_identity
                and str(previous.get("id")) == item_id
                and str(previous.get("role")) == role
            )
            if not same_claim:
                raise FileExistsError(
                    f"Final row {item_id} was already exposed by "
                    f"{previous.get('confirmation_run', marker)}"
                ) from error
            existing_records.append(previous)

    claim = {
        "claimed_at": created_at,
        "claim_identity": claim_identity,
        "confirmation_run": str(Path(confirmation_run).resolve()),
        "selection_lock_sha256": selection_lock_sha256,
        "roles": {
            role: {
                "count": len(rows),
                "row_id_set_sha256": row_id_set_sha256(rows),
                "file_sha256": str(locked_files[role]["sha256"]),
            }
            for role, rows in sorted(materialized.items())
        },
        "row_marker_count": len(all_ids),
        "idempotent_resume": bool(existing_records),
        "active_lease": lease,
        "preexisting_marker_count": len(existing_records),
        "first_confirmation_runs": sorted(
            {
                str(record.get("confirmation_run"))
                for record in existing_records
                if record.get("confirmation_run")
            }
        ),
    }
    canonical_claim_path = claim_root / f"{claim_identity}.json"
    if canonical_claim_path.exists():
        previous_claim = read_json(canonical_claim_path)
        if (
            str(previous_claim.get("claim_identity")) != claim_identity
            or str(previous_claim.get("selection_lock_sha256"))
            != selection_lock_sha256
        ):
            raise ValueError("Existing final claim record conflicts with this lock")
    else:
        write_json(canonical_claim_path, claim)
    return claim


def record_final_claim_completion(
    config: dict[str, Any],
    *,
    claim_identity: str,
    selection_lock_sha256: str,
    confirmation_run: str | Path,
) -> dict[str, Any]:
    """Create an immutable registry marker after a confirmation fully succeeds."""

    run = Path(confirmation_run).resolve()
    manifest = read_json(run / "run_manifest.json")
    if (
        manifest.get("phase") != "stage4_confirmation"
        or manifest.get("status") != "completed"
        or str(manifest.get("selection_lock_sha256")) != selection_lock_sha256
        or not (run / "SUCCESS").is_file()
    ):
        raise ValueError("Cannot finalize a Stage 4 claim before confirmation succeeds")
    runs_raw = Path(config["paths"]["runs_dir"])
    runs_root = runs_raw if runs_raw.is_absolute() else Path(config["_project_root"]) / runs_raw
    registry = runs_root.resolve() / ".stage4_final_exposures"
    claim_path = registry / "claims" / f"{claim_identity}.json"
    claim = read_json(claim_path)
    if str(claim.get("selection_lock_sha256")) != selection_lock_sha256:
        raise ValueError("Final claim and successful confirmation lock differ")
    completed_root = registry / "completed"
    completed_root.mkdir(parents=True, exist_ok=True)
    marker = completed_root / f"{claim_identity}.json"
    record = {
        "claim_identity": claim_identity,
        "completed_at": _utc_now(),
        "confirmation_run": str(run),
        "selection_lock_sha256": selection_lock_sha256,
    }
    try:
        with marker.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        previous = read_json(marker)
        if (
            str(previous.get("claim_identity")) != claim_identity
            or str(previous.get("selection_lock_sha256")) != selection_lock_sha256
            or str(previous.get("confirmation_run")) != str(run)
        ):
            raise ValueError("Existing completion marker conflicts with this confirmation")
        return previous
    return record
