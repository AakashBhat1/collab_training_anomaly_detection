from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ClipItem:
    clip_id: str
    class_name: str
    source_path: str


def discover_clips(source_dir: str | Path, classes: list[str]) -> list[ClipItem]:
    source_root = Path(source_dir)
    items: list[ClipItem] = []

    for class_name in classes:
        class_dir = source_root / class_name
        if not class_dir.exists():
            continue

        for entry in sorted(class_dir.iterdir()):
            if entry.is_dir():
                items.append(
                    ClipItem(
                        clip_id=f"{class_name}:{entry.name}",
                        class_name=class_name,
                        source_path=str(entry),
                    )
                )
                continue

            if entry.suffix.lower() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
                items.append(
                    ClipItem(
                        clip_id=f"{class_name}:{entry.stem}",
                        class_name=class_name,
                        source_path=str(entry),
                    )
                )

    return items


def _validate_splits(splits: dict[str, float]) -> None:
    required = {"train", "val", "test"}
    if set(splits.keys()) != required:
        raise ValueError("splits must include exactly train/val/test keys")

    for split_name in ("train", "val", "test"):
        value = splits[split_name]
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"split ratio for {split_name} must be between 0.0 and 1.0")

    if abs(sum(splits.values()) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")


def plan_split(
    items: list[ClipItem],
    splits: dict[str, float],
    seed: int = 42,
) -> dict[str, list[ClipItem]]:
    _validate_splits(splits)

    rng = random.Random(seed)
    by_class: dict[str, list[ClipItem]] = {}
    for item in items:
        by_class.setdefault(item.class_name, []).append(item)

    output: dict[str, list[ClipItem]] = {"train": [], "val": [], "test": []}

    for class_name, class_items in by_class.items():
        shuffled = list(class_items)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = int(total * splits["train"])
        val_end = train_end + int(total * splits["val"])

        output["train"].extend(shuffled[:train_end])
        output["val"].extend(shuffled[train_end:val_end])
        output["test"].extend(shuffled[val_end:])

        # Preserve deterministic ordering by class then clip id.
        output["train"].sort(key=lambda x: (x.class_name, x.clip_id))
        output["val"].sort(key=lambda x: (x.class_name, x.clip_id))
        output["test"].sort(key=lambda x: (x.class_name, x.clip_id))

    _assert_no_overlap(output)
    return output


def _assert_no_overlap(plan: dict[str, list[ClipItem]]) -> None:
    seen: set[str] = set()
    for split_name in ("train", "val", "test"):
        for clip in plan.get(split_name, []):
            if clip.clip_id in seen:
                raise ValueError(f"Clip overlap detected for clip_id={clip.clip_id}")
            seen.add(clip.clip_id)


def write_split_manifest(path: str | Path, plan: dict[str, list[ClipItem]]) -> None:
    serializable = {key: [asdict(item) for item in value] for key, value in plan.items()}
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def extract_mp4_to_frames(source_video: Path, target_dir: Path) -> None:
    import cv2

    target_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video clip: {source_video}")

    frame_index = 1
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_path = target_dir / f"{frame_index:06d}.jpg"
        wrote = cv2.imwrite(str(frame_path), frame)
        if not wrote:
            capture.release()
            raise ValueError(f"Unable to write extracted frame: {frame_path}")
        frame_index += 1

    capture.release()
    if frame_index == 1:
        raise ValueError(f"Video clip contains no frames: {source_video}")


def _clip_dir_name(item: ClipItem, source: Path) -> str:
    if ":" in item.clip_id:
        return item.clip_id.split(":", maxsplit=1)[1]
    if source.is_file():
        return source.stem
    return source.name


def materialize_split_plan(
    plan: dict[str, list[ClipItem]],
    output_dir: str | Path,
) -> None:
    required_splits = {"train", "val", "test"}
    if set(plan.keys()) != required_splits:
        raise ValueError("split plan must include exactly train/val/test keys")

    target_root = Path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split_dir = target_root / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for split_name in ("train", "val", "test"):
        items = plan[split_name]
        for item in items:
            source = Path(item.source_path)
            destination = target_root / split_name / item.class_name / _clip_dir_name(item, source)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                suffix = source.suffix.lower()
                if suffix in VIDEO_EXTENSIONS:
                    extract_mp4_to_frames(source, destination)
                elif suffix in IMAGE_EXTENSIONS:
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination / f"000001{suffix}")
                else:
                    raise ValueError(f"Unsupported clip file type: {source}")

