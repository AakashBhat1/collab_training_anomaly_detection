from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class ClipRecord:
    class_name: str
    frame_dir: Path


def _build_transform(image_size: int, train: bool) -> transforms.Compose:
    transform_steps: list[transforms.Compose | transforms.Normalize | transforms.Resize] = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(transform_steps)


class ActionClipDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        split_dir: str | Path,
        classes: list[str],
        sequence_length: int,
        image_size: int,
        train: bool,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.classes = classes
        self.sequence_length = sequence_length
        self.class_to_index = {name: idx for idx, name in enumerate(classes)}
        self.transform = _build_transform(image_size=image_size, train=train)
        self.records = self._discover_records()

        if not self.records:
            raise ValueError(f"No clips discovered under {self.split_dir}")

    def _discover_records(self) -> list[ClipRecord]:
        records: list[ClipRecord] = []
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for clip_dir in sorted(class_dir.iterdir()):
                if clip_dir.is_dir():
                    records.append(ClipRecord(class_name=class_name, frame_dir=clip_dir))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _select_frames(self, frame_paths: list[Path]) -> list[Path]:
        if len(frame_paths) >= self.sequence_length:
            stride = len(frame_paths) / float(self.sequence_length)
            return [frame_paths[int(i * stride)] for i in range(self.sequence_length)]

        padding = [frame_paths[-1]] * (self.sequence_length - len(frame_paths))
        return frame_paths + padding

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        frame_paths = sorted(
            [
                path
                for path in record.frame_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if not frame_paths:
            raise ValueError(f"Clip contains no image frames: {record.frame_dir}")

        sampled = self._select_frames(frame_paths)
        frames: list[torch.Tensor] = []
        for frame_path in sampled:
            image = Image.open(frame_path).convert("RGB")
            frames.append(self.transform(image))

        clip_tensor = torch.stack(frames, dim=0)
        label = self.class_to_index[record.class_name]
        return clip_tensor, label

