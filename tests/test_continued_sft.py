import json
from pathlib import Path

import pytest
import torch

from reasoning_lab.training.continued_sft import (
    _continuation_dataset,
    _disable_dropout_modules,
    _require_fresh_output,
    _train_from_fresh_state,
    sha256_directory,
    validate_parent_lineage,
)


REVISION = "a" * 40


def test_control_dropout_can_be_disabled_deterministically() -> None:
    model = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.Linear(2, 2))
    assert _disable_dropout_modules(model) == [0.0]
    assert model[0].p == 0.0


def _write_adapter(path: Path, *, revision: str = REVISION) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-1.7B-Base",
                "revision": revision,
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter-weights")


def test_directory_hash_is_relocatable_and_content_sensitive(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_adapter(left)
    _write_adapter(right)

    assert sha256_directory(left) == sha256_directory(right)
    (right / "adapter_model.safetensors").write_bytes(b"changed")
    assert sha256_directory(left) != sha256_directory(right)


def test_parent_lineage_requires_matching_immutable_revision(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter(adapter)

    lineage = validate_parent_lineage(
        adapter,
        model_id="Qwen/Qwen3-1.7B-Base",
        revision=REVISION,
    )
    assert lineage["base_model_revision"] == REVISION
    assert len(lineage["adapter_sha256"]) == 64

    with pytest.raises(ValueError, match="immutable 40-character"):
        validate_parent_lineage(
            adapter,
            model_id="Qwen/Qwen3-1.7B-Base",
            revision="main",
        )
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_parent_lineage(
            adapter,
            model_id="Qwen/Qwen3-1.7B-Base",
            revision="b" * 40,
        )


def test_fresh_branch_rejects_existing_training_state(tmp_path: Path) -> None:
    output = tmp_path / "branch"
    _, checkpoints, final_adapter = _require_fresh_output(output)
    assert checkpoints.is_dir()
    assert final_adapter == output / "final_adapter"

    with pytest.raises(FileExistsError, match="never resumes"):
        _require_fresh_output(output)


def test_continuation_dataset_keeps_only_training_contract(tmp_path: Path) -> None:
    path = tmp_path / "continuation.jsonl"
    path.write_text(
        json.dumps({"prompt": "2 + 2 =", "completion": "4", "id": "ignored"}) + "\n",
        encoding="utf-8",
    )
    dataset = _continuation_dataset(path)
    assert dataset.column_names == ["prompt", "completion"]
    assert dataset[0] == {"prompt": "2 + 2 =", "completion": "4"}


def test_fresh_training_passes_no_checkpoint_or_optimizer_state() -> None:
    class State:
        global_step = 0

    class Trainer:
        state = State()
        optimizer = None
        lr_scheduler = None

        def __init__(self) -> None:
            self.kwargs = None

        def train(self, **kwargs):
            self.kwargs = kwargs
            return "result"

    trainer = Trainer()
    assert _train_from_fresh_state(trainer) == "result"
    assert trainer.kwargs == {"resume_from_checkpoint": None}
