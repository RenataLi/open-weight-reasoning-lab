from pathlib import Path

import pytest

from reasoning_lab.config import load_config


def test_smoke_config_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "e2e_smoke.yaml")
    assert config["model"]["id"] == "Qwen/Qwen3-0.6B-Base"
    assert len(config["_config_hash"]) == 64


def test_stage2_config_pins_model_and_external_dataset() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "stage2_qwen3_1_7b.yaml")
    assert config["model"]["id"] == "Qwen/Qwen3-1.7B-Base"
    assert len(config["model"]["revision"]) == 40
    assert len(config["external_evaluation"]["revision"]) == 40
    assert config["data"]["continuation_size"] == 256


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "e2e_smoke.yaml"
    text = source.read_text(encoding="utf-8") + "\nunknown: true\n"
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
