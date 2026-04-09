from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class ArtifactFilenames:
    pt: str
    onnx: str
    xml: str
    bin: str


def build_artifact_filenames(
    model_version: str,
    date_str: str | None = None,
    base_name: str = "action_model",
) -> ArtifactFilenames:
    safe_date = date_str or date.today().isoformat()
    prefix = f"{base_name}_v{model_version}_{safe_date}"
    return ArtifactFilenames(
        pt=f"{prefix}.pt",
        onnx=f"{prefix}.onnx",
        xml=f"{prefix}.xml",
        bin=f"{prefix}.bin",
    )


def build_artifact_paths(
    output_dir: str | Path,
    model_version: str,
    date_str: str | None = None,
    base_name: str = "action_model",
) -> dict[str, Path]:
    names = build_artifact_filenames(
        model_version=model_version,
        date_str=date_str,
        base_name=base_name,
    )
    artifact_dir = Path(output_dir)
    return {
        "pt": artifact_dir / names.pt,
        "onnx": artifact_dir / names.onnx,
        "xml": artifact_dir / names.xml,
        "bin": artifact_dir / names.bin,
    }

