from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from reasoning_lab.data import svamp


def _row(row_id: str, question: str, family: str = "svamp_addition") -> dict[str, str]:
    return {"id": row_id, "question": question, "family": family}


def test_forced_validation_keeps_every_other_row_in_development_quarantine() -> None:
    rows = [
        _row("svamp_a", "Alice has 12 red balls and buys 5 more. How many now?"),
        _row("svamp_b", "Alice has 90 red balls and buys 7 more. How many now?"),
        _row("svamp_c", "A train travels east while a clock reports noon."),
    ]

    validation, quarantine, diagnostics = (
        svamp._split_development_quarantine(
            copy.deepcopy(rows),
            forced_validation_ids={"svamp_a"},
        )
    )

    assert [row["id"] for row in validation] == ["svamp_a"]
    assert [row["id"] for row in quarantine] == ["svamp_b", "svamp_c"]
    assert diagnostics["forced_validation_matched_count"] == 1
    assert diagnostics["development_quarantine_count"] == 2
    assert diagnostics["all_verified_rows_development_exposed"] is True
    assert diagnostics["confirmation_count"] == 0


def test_ensure_source_downloads_only_missing_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned-svamp-test-payload"
    monkeypatch.setattr(svamp, "SVAMP_BYTES", len(payload))
    monkeypatch.setattr(svamp, "SVAMP_SHA256", hashlib.sha256(payload).hexdigest())
    calls: list[int] = []

    def download(timeout_seconds: int = 60) -> bytes:
        calls.append(timeout_seconds)
        return payload

    monkeypatch.setattr(svamp, "_download_svamp_bytes", download)
    destination = tmp_path / "source" / "SVAMP.json"

    assert svamp.ensure_svamp_source(destination, timeout_seconds=13) == destination.resolve()
    assert destination.read_bytes() == payload
    assert calls == [13]
    assert svamp.ensure_svamp_source(destination) == destination.resolve()
    assert calls == [13]


def test_ensure_source_rejects_unpinned_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(svamp, "SVAMP_BYTES", 3)
    monkeypatch.setattr(svamp, "SVAMP_SHA256", hashlib.sha256(b"yes").hexdigest())
    monkeypatch.setattr(svamp, "_download_svamp_bytes", lambda timeout_seconds=60: b"bad")
    destination = tmp_path / "SVAMP.json"

    with pytest.raises(ValueError, match="checksum"):
        svamp.ensure_svamp_source(destination)
    assert not destination.exists()
