from pathlib import Path

from reasoning_lab.training.sft import latest_checkpoint


def test_latest_checkpoint_uses_numeric_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"

