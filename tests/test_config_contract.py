from __future__ import annotations

import json
from pathlib import Path

import pytest

from collab_scripts.config_schema import PipelineConfig, load_pipeline_config


def test_load_pipeline_config_parses_expected_classes(tmp_path: Path) -> None:
    config_data = {
        "classes": ["fight", "theft", "intrusion", "normal"],
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 16,
        "crop_size": 224,
        "batch_size": 8,
        "epochs": 30,
        "learning_rate": 1e-4,
        "confidence_threshold": 0.6,
        "smoothing_window": 5,
        "n_of_m": {"n": 3, "m": 5},
    }
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    config = load_pipeline_config(config_path)

    assert isinstance(config, PipelineConfig)
    assert config.classes == ["fight", "theft", "intrusion", "normal"]
    assert config.splits == {"train": 0.7, "val": 0.15, "test": 0.15}


def test_load_pipeline_config_rejects_invalid_split_sum(tmp_path: Path) -> None:
    bad_config = {
        "classes": ["fight", "theft", "intrusion", "normal"],
        "splits": {"train": 0.7, "val": 0.2, "test": 0.2},
        "sequence_length": 16,
        "crop_size": 224,
        "batch_size": 8,
        "epochs": 30,
        "learning_rate": 1e-4,
        "confidence_threshold": 0.6,
        "smoothing_window": 5,
        "n_of_m": {"n": 3, "m": 5},
    }
    config_path = tmp_path / "bad_pipeline_config.json"
    config_path.write_text(json.dumps(bad_config), encoding="utf-8")

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_pipeline_config(config_path)
