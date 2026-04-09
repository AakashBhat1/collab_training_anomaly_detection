from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_CLASSES = ("fight", "theft", "intrusion", "normal")


@dataclass(frozen=True)
class NOfMConfig:
    n: int
    m: int


@dataclass(frozen=True)
class PathConfig:
    raw_dataset_dir: str
    dataset_dir: str
    checkpoint_dir: str
    drive_checkpoint_dir: str
    artifact_dir: str


@dataclass(frozen=True)
class PipelineConfig:
    classes: list[str]
    splits: dict[str, float]
    sequence_length: int
    crop_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    confidence_threshold: float
    smoothing_window: int
    n_of_m: NOfMConfig
    paths: PathConfig | None = None
    model_version: str = "0.1.0"

    def validate(self) -> None:
        missing = [name for name in REQUIRED_CLASSES if name not in self.classes]
        if missing:
            raise ValueError(f"Config classes missing required values: {missing}")

        split_keys = {"train", "val", "test"}
        if set(self.splits.keys()) != split_keys:
            raise ValueError("Config splits must contain exactly train/val/test keys")

        split_sum = sum(self.splits.values())
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.n_of_m.n <= 0 or self.n_of_m.m <= 0:
            raise ValueError("n_of_m values must be positive")
        if self.n_of_m.n > self.n_of_m.m:
            raise ValueError("n_of_m.n must be <= n_of_m.m")


def _parse_paths(value: Any) -> PathConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("paths must be an object when provided")
    return PathConfig(
        raw_dataset_dir=str(value["raw_dataset_dir"]),
        dataset_dir=str(value["dataset_dir"]),
        checkpoint_dir=str(value["checkpoint_dir"]),
        drive_checkpoint_dir=str(value["drive_checkpoint_dir"]),
        artifact_dir=str(value["artifact_dir"]),
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw_data = json.loads(config_path.read_text(encoding="utf-8"))

    config = PipelineConfig(
        classes=list(raw_data["classes"]),
        splits={k: float(v) for k, v in raw_data["splits"].items()},
        sequence_length=int(raw_data["sequence_length"]),
        crop_size=int(raw_data["crop_size"]),
        batch_size=int(raw_data["batch_size"]),
        epochs=int(raw_data["epochs"]),
        learning_rate=float(raw_data["learning_rate"]),
        confidence_threshold=float(raw_data["confidence_threshold"]),
        smoothing_window=int(raw_data["smoothing_window"]),
        n_of_m=NOfMConfig(
            n=int(raw_data["n_of_m"]["n"]),
            m=int(raw_data["n_of_m"]["m"]),
        ),
        paths=_parse_paths(raw_data.get("paths")),
        model_version=str(raw_data.get("model_version", "0.1.0")),
    )
    config.validate()
    return config


def save_pipeline_config(path: str | Path, config: PipelineConfig) -> None:
    payload = {
        "classes": config.classes,
        "splits": config.splits,
        "sequence_length": config.sequence_length,
        "crop_size": config.crop_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "confidence_threshold": config.confidence_threshold,
        "smoothing_window": config.smoothing_window,
        "n_of_m": {"n": config.n_of_m.n, "m": config.n_of_m.m},
        "model_version": config.model_version,
    }
    if config.paths is not None:
        payload["paths"] = {
            "raw_dataset_dir": config.paths.raw_dataset_dir,
            "dataset_dir": config.paths.dataset_dir,
            "checkpoint_dir": config.paths.checkpoint_dir,
            "drive_checkpoint_dir": config.paths.drive_checkpoint_dir,
            "artifact_dir": config.paths.artifact_dir,
        }

    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

