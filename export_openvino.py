from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import torch

from collab_scripts.artifacts import build_artifact_paths
from collab_scripts.config_schema import load_pipeline_config
from collab_scripts.model import CNNLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained checkpoint to ONNX and OpenVINO IR.")
    parser.add_argument(
        "--config",
        default="collab_scripts/pipeline_config.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint filename under checkpoint_dir (default: best.pt).",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Artifact date suffix (YYYY-MM-DD).",
    )
    return parser.parse_args()


def _export_onnx(model: CNNLSTM, checkpoint_path: Path, onnx_path: Path, seq_len: int, crop_size: int) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, seq_len, 3, crop_size, crop_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def _convert_openvino(onnx_path: Path, xml_path: Path) -> None:
    import openvino as ov
    from openvino.tools import mo

    ov_model = mo.convert_model(str(onnx_path))
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(xml_path))


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    if config.paths is None:
        raise ValueError("`paths` must be defined in pipeline config for export.")

    checkpoint_path = Path(config.paths.checkpoint_dir) / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    artifact_paths = build_artifact_paths(
        output_dir=config.paths.artifact_dir,
        model_version=config.model_version,
        date_str=args.date,
    )

    model = CNNLSTM(num_classes=len(config.classes))
    _export_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        onnx_path=artifact_paths["onnx"],
        seq_len=config.sequence_length,
        crop_size=config.crop_size,
    )
    _convert_openvino(onnx_path=artifact_paths["onnx"], xml_path=artifact_paths["xml"])

    print(f"ONNX artifact: {artifact_paths['onnx']}")
    print(f"OpenVINO XML: {artifact_paths['xml']}")
    print(f"OpenVINO BIN: {artifact_paths['bin']}")


if __name__ == "__main__":
    main()

