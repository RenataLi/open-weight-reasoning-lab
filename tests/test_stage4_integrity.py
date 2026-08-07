from __future__ import annotations

import hashlib
import json

import pytest

import reasoning_lab.experiment_integrity as integrity
from reasoning_lab.experiment_integrity import (
    attest_stage4_implementation,
    claim_final_rows,
    record_final_claim_completion,
    row_id_set_sha256,
)
from reasoning_lab.io_utils import sha256_file


def test_stage4_attestation_covers_metric_parsing_code() -> None:
    assert "src/reasoning_lab/evaluation/parsing.py" in integrity.STAGE4_IMPLEMENTATION_FILES


def _locked(tmp_path, role: str, ids: list[str]) -> dict:
    path = tmp_path / f"{role}.jsonl"
    rows = [{"id": item_id} for item_id in ids]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return {
        "role": role,
        "path": str(path),
        "count": len(rows),
        "sha256": sha256_file(path),
        "row_id_set_sha256": row_id_set_sha256(rows),
    }


def _config(tmp_path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "paths": {"runs_dir": "runs"},
    }


def test_final_exposure_blocks_partial_overlap(tmp_path) -> None:
    config = _config(tmp_path)
    first = {"final_a": _locked(tmp_path, "final_a", ["x", "y"])}
    claim = claim_final_rows(
        config,
        locked_files=first,
        selection_lock_sha256="a" * 64,
        confirmation_run=tmp_path / "run1",
    )
    assert claim["row_marker_count"] == 2

    second = {"final_b": _locked(tmp_path, "final_b", ["y", "z"])}
    with pytest.raises(FileExistsError, match="already exposed"):
        claim_final_rows(
            config,
            locked_files=second,
            selection_lock_sha256="b" * 64,
            confirmation_run=tmp_path / "run2",
        )


@pytest.mark.parametrize("terminal_status", ["failed", "aborted"])
def test_final_exposure_is_idempotent_for_the_same_locked_selection(
    tmp_path, terminal_status: str
) -> None:
    config = _config(tmp_path)
    locked = {"final_a": _locked(tmp_path, "final_a", ["x", "y"])}
    first_run = tmp_path / "run1"
    first = claim_final_rows(
        config,
        locked_files=locked,
        selection_lock_sha256="a" * 64,
        confirmation_run=first_run,
    )
    first_run.mkdir()
    (first_run / "run_manifest.json").write_text(
        json.dumps({"phase": "stage4_confirmation", "status": terminal_status}),
        encoding="utf-8",
    )
    resumed = claim_final_rows(
        config,
        locked_files=locked,
        selection_lock_sha256="a" * 64,
        confirmation_run=tmp_path / "run2",
    )
    assert not first["idempotent_resume"]
    assert resumed["idempotent_resume"]
    assert resumed["preexisting_marker_count"] == 2


def test_active_confirmation_lease_blocks_parallel_same_lock(tmp_path) -> None:
    config = _config(tmp_path)
    locked = {"final_a": _locked(tmp_path, "final_a", ["x", "y"])}
    first_run = tmp_path / "run1"
    first_run.mkdir()
    (first_run / "run_manifest.json").write_text(
        json.dumps({"phase": "stage4_confirmation", "status": "running"}),
        encoding="utf-8",
    )
    claim_final_rows(
        config,
        locked_files=locked,
        selection_lock_sha256="a" * 64,
        confirmation_run=first_run,
    )
    with pytest.raises(FileExistsError, match="already owns"):
        claim_final_rows(
            config,
            locked_files=locked,
            selection_lock_sha256="a" * 64,
            confirmation_run=tmp_path / "run2",
        )


def test_successful_confirmation_permanently_closes_same_lock(tmp_path) -> None:
    config = _config(tmp_path)
    locked = {"final_a": _locked(tmp_path, "final_a", ["x", "y"])}
    run = tmp_path / "runs" / "confirmation"
    first = claim_final_rows(
        config,
        locked_files=locked,
        selection_lock_sha256="a" * 64,
        confirmation_run=run,
    )
    run.mkdir(parents=True)
    (run / "SUCCESS").write_text("completed\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "phase": "stage4_confirmation",
                "status": "completed",
                "selection_lock_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    record_final_claim_completion(
        config,
        claim_identity=first["claim_identity"],
        selection_lock_sha256="a" * 64,
        confirmation_run=run,
    )
    with pytest.raises(FileExistsError, match="successful one-time confirmation"):
        claim_final_rows(
            config,
            locked_files=locked,
            selection_lock_sha256="a" * 64,
            confirmation_run=tmp_path / "runs" / "retry",
        )


def test_final_exposure_detects_tamper_and_role_rename(tmp_path) -> None:
    config = _config(tmp_path)
    item = _locked(tmp_path, "final_a", ["x"])
    renamed = dict(item)
    renamed["role"] = "anything"
    with pytest.raises(ValueError, match="role mismatch"):
        claim_final_rows(
            config,
            locked_files={"final_a": renamed},
            selection_lock_sha256="a" * 64,
            confirmation_run=tmp_path / "run",
        )

    path = tmp_path / "final_a.jsonl"
    path.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        claim_final_rows(
            config,
            locked_files={"final_a": item},
            selection_lock_sha256="a" * 64,
            confirmation_run=tmp_path / "run",
        )


def test_stage4_implementation_attestation_changes_on_source_tamper(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "stage4.yaml"
    config_path.parent.mkdir()
    config_path.write_text("stage4: {}\n", encoding="utf-8")
    monkeypatch.setattr(integrity, "STAGE4_IMPLEMENTATION_FILES", ("source.py",))
    config = {
        "_project_root": str(tmp_path),
        "_config_path": str(config_path),
    }
    before = attest_stage4_implementation(config)
    assert set(before["source_files"]) == {"configs/stage4.yaml", "source.py"}
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = attest_stage4_implementation(config)
    assert before["combined_sha256"] != after["combined_sha256"]
