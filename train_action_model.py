from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from collab_scripts.config_schema import PipelineConfig, load_pipeline_config
from collab_scripts.model import CNNLSTM
from collab_scripts.training_data import ActionClipDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+LSTM action classifier.")
    parser.add_argument(
        "--config",
        default="collab_scripts/pipeline_config.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from last checkpoint if available.",
    )
    return parser.parse_args()


def _sync_checkpoint_to_drive(local_file: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    if local_file.exists():
        shutil.copy2(local_file, drive_dir / local_file.name)


def _load_resume_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    local_last: Path,
    drive_last: Path,
) -> tuple[int, float]:
    if not local_last.exists() and drive_last.exists():
        local_last.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_last, local_last)

    if not local_last.exists():
        return 0, 0.0

    checkpoint = torch.load(local_last, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_val_recall"])


def _calculate_class_weights(dataset: ActionClipDataset) -> torch.Tensor:
    counts = Counter(record.class_name for record in dataset.records)
    weights = []
    for class_name in dataset.classes:
        count = max(1, counts.get(class_name, 0))
        weights.append(1.0 / float(count))
    return torch.tensor(weights, dtype=torch.float32)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(clips)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(1, total_batches)


def _validate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)

            logits = model(clips)
            loss = loss_fn(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            total_loss += float(loss.item())
            total_batches += 1

    macro_recall = float(recall_score(all_labels, all_predictions, average="macro", zero_division=0))
    return total_loss / max(1, total_batches), macro_recall


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_recall: float,
    config: PipelineConfig,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_val_recall": best_val_recall,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "classes": config.classes,
        "sequence_length": config.sequence_length,
        "crop_size": config.crop_size,
        "model_version": config.model_version,
    }
    torch.save(payload, checkpoint_path)


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    if config.paths is None:
        raise ValueError("`paths` must be defined in pipeline config for training.")

    dataset_dir = Path(config.paths.dataset_dir)
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    drive_checkpoint_dir = Path(config.paths.drive_checkpoint_dir)

    train_dataset = ActionClipDataset(
        split_dir=dataset_dir / "train",
        classes=config.classes,
        sequence_length=config.sequence_length,
        image_size=config.crop_size,
        train=True,
    )
    val_dataset = ActionClipDataset(
        split_dir=dataset_dir / "val",
        classes=config.classes,
        sequence_length=config.sequence_length,
        image_size=config.crop_size,
        train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(num_classes=len(config.classes))
    model.to(device)

    class_weights = _calculate_class_weights(train_dataset).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    local_last = checkpoint_dir / "last.pt"
    local_best = checkpoint_dir / "best.pt"
    drive_last = drive_checkpoint_dir / "last.pt"

    start_epoch = 0
    best_val_recall = 0.0
    if args.auto_resume:
        start_epoch, best_val_recall = _load_resume_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            local_last=local_last,
            drive_last=drive_last,
        )
        print(f"Auto-resume enabled. Starting at epoch: {start_epoch}")

    history: list[dict[str, float | int]] = []

    for epoch in range(start_epoch, config.epochs):
        train_loss = _train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_macro_recall = _validate(model, val_loader, loss_fn, device)
        scheduler.step(val_macro_recall)

        _save_checkpoint(
            checkpoint_path=local_last,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_recall=best_val_recall,
            config=config,
        )
        _sync_checkpoint_to_drive(local_last, drive_checkpoint_dir)

        if val_macro_recall >= best_val_recall:
            best_val_recall = val_macro_recall
            _save_checkpoint(
                checkpoint_path=local_best,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_recall=best_val_recall,
                config=config,
            )
            _sync_checkpoint_to_drive(local_best, drive_checkpoint_dir)

        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_macro_recall": round(val_macro_recall, 6),
            "best_val_recall": round(best_val_recall, 6),
        }
        history.append(epoch_row)
        print(epoch_row)

    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()

