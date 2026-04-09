from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from torch.utils.data import DataLoader

from collab_scripts.config_schema import load_pipeline_config
from collab_scripts.device import get_device, is_xla_device
from collab_scripts.model import CNNLSTM
from collab_scripts.training_data import ActionClipDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action classifier on held-out test split.")
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
        "--output",
        default="evaluation_report.json",
        help="Output JSON filename under artifact_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    if config.paths is None:
        raise ValueError("`paths` must be defined in pipeline config for evaluation.")

    checkpoint_path = Path(config.paths.checkpoint_dir) / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    loader_workers = 0 if is_xla_device(device) else 2

    test_dataset = ActionClipDataset(
        split_dir=Path(config.paths.dataset_dir) / "test",
        classes=config.classes,
        sequence_length=config.sequence_length,
        image_size=config.crop_size,
        train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=loader_workers)

    model = CNNLSTM(num_classes=len(config.classes))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    labels: list[int] = []
    preds: list[int] = []

    with torch.no_grad():
        for clips, targets in test_loader:
            clips = clips.to(device)
            logits = model(clips)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(predictions)
            labels.extend(targets.tolist())

    macro_recall = float(recall_score(labels, preds, average="macro", zero_division=0))
    per_class_report = classification_report(
        labels,
        preds,
        target_names=config.classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, preds).tolist()

    output_dir = Path(config.paths.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output
    report = {
        "checkpoint": str(checkpoint_path),
        "model_version": config.model_version,
        "macro_recall": macro_recall,
        "class_report": per_class_report,
        "confusion_matrix": matrix,
        "classes": config.classes,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation report written to: {output_path}")


if __name__ == "__main__":
    main()

