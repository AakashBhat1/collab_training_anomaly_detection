from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import collab_scripts.dataset_split as dataset_split
import pytest

from collab_scripts.dataset_split import ClipItem, plan_split


def _sample_items() -> list[ClipItem]:
    items: list[ClipItem] = []
    for class_name in ("fight", "theft", "intrusion", "normal"):
        for index in range(10):
            items.append(
                ClipItem(
                    clip_id=f"{class_name}_clip_{index:02d}",
                    class_name=class_name,
                    source_path=f"/tmp/{class_name}/{index:02d}.mp4",
                )
            )
    return items


def test_split_planner_is_seed_deterministic() -> None:
    items = _sample_items()
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}

    plan_a = plan_split(items=items, splits=splits, seed=42)
    plan_b = plan_split(items=items, splits=splits, seed=42)

    assert plan_a == plan_b


def test_split_planner_has_no_clip_overlap() -> None:
    items = _sample_items()
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    plan = plan_split(items=items, splits=splits, seed=7)

    seen: set[str] = set()
    for split_items in plan.values():
        for clip in split_items:
            assert clip.clip_id not in seen
            seen.add(clip.clip_id)


def test_materialize_split_plan_converts_mp4_into_frame_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_extract_mp4_to_frames(source_video: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "000001.jpg").write_bytes(b"f1")
        (target_dir / "000002.jpg").write_bytes(b"f2")
        calls.append((source_video, target_dir))

    monkeypatch.setattr(dataset_split, "extract_mp4_to_frames", fake_extract_mp4_to_frames)

    source_mp4 = tmp_path / "raw" / "fight" / "clip_001.mp4"
    source_mp4.parent.mkdir(parents=True, exist_ok=True)
    source_mp4.write_bytes(b"fake-mp4")

    plan = {
        "train": [
            ClipItem(
                clip_id="fight:clip_001",
                class_name="fight",
                source_path=str(source_mp4),
            )
        ],
        "val": [],
        "test": [],
    }

    output_dir = tmp_path / "dataset"
    dataset_split.materialize_split_plan(plan=plan, output_dir=output_dir)

    clip_dir = output_dir / "train" / "fight" / "clip_001"
    assert clip_dir.is_dir()
    assert sorted(path.name for path in clip_dir.glob("*.jpg")) == ["000001.jpg", "000002.jpg"]
    assert not (output_dir / "train" / "fight" / "clip_001.mp4").exists()
    assert calls == [(source_mp4, clip_dir)]


def test_materialize_split_plan_cleans_stale_split_directories_on_rematerialization(
    tmp_path: Path,
) -> None:
    old_clip = tmp_path / "raw" / "fight" / "clip_old"
    old_clip.mkdir(parents=True, exist_ok=True)
    (old_clip / "000001.jpg").write_bytes(b"a")

    val_old_clip = tmp_path / "raw" / "theft" / "clip_val_old"
    val_old_clip.mkdir(parents=True, exist_ok=True)
    (val_old_clip / "000001.jpg").write_bytes(b"b")

    new_clip = tmp_path / "raw" / "fight" / "clip_new"
    new_clip.mkdir(parents=True, exist_ok=True)
    (new_clip / "000001.jpg").write_bytes(b"c")

    output_dir = tmp_path / "dataset"
    plan_v1 = {
        "train": [ClipItem("fight:clip_old", "fight", str(old_clip))],
        "val": [ClipItem("theft:clip_val_old", "theft", str(val_old_clip))],
        "test": [],
    }
    dataset_split.materialize_split_plan(plan=plan_v1, output_dir=output_dir)

    plan_v2 = {
        "train": [ClipItem("fight:clip_new", "fight", str(new_clip))],
        "val": [],
        "test": [],
    }
    dataset_split.materialize_split_plan(plan=plan_v2, output_dir=output_dir)

    actual_clip_dirs = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.glob("*/*/*")
        if path.is_dir()
    }

    assert actual_clip_dirs == {"train/fight/clip_new"}
    assert not (output_dir / "train" / "fight" / "clip_old").exists()
    assert not (output_dir / "val").exists()


def test_discover_clips_collects_frame_dirs_and_media_files(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    frame_dir = root / "fight" / "clip_a"
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / "0001.jpg").write_bytes(b"img")
    video_file = root / "fight" / "clip_b.mp4"
    video_file.write_bytes(b"video")
    image_file = root / "theft" / "still_1.jpg"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.write_bytes(b"img")

    clips = dataset_split.discover_clips(root, classes=["fight", "theft", "intrusion", "normal"])

    clip_ids = {item.clip_id for item in clips}
    assert clip_ids == {"fight:clip_a", "fight:clip_b", "theft:still_1"}


def test_plan_split_rejects_bad_keys() -> None:
    with pytest.raises(ValueError, match="train/val/test"):
        plan_split(items=[], splits={"train": 0.8, "val": 0.2})


def test_plan_split_rejects_bad_ratio_sum() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        plan_split(items=[], splits={"train": 0.8, "val": 0.1, "test": 0.2})


def test_plan_split_rejects_ratio_values_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        plan_split(items=[], splits={"train": 1.1, "val": 0.0, "test": -0.1})


def test_plan_split_rejects_negative_ratio_even_when_sum_is_one() -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        plan_split(items=[], splits={"train": -0.1, "val": 0.6, "test": 0.5})


def test_write_split_manifest_serializes_clip_items(tmp_path: Path) -> None:
    manifest_path = tmp_path / "split_manifest.json"
    plan = {
        "train": [ClipItem("fight:clip_1", "fight", "/tmp/fight/clip_1")],
        "val": [],
        "test": [],
    }

    dataset_split.write_split_manifest(manifest_path, plan)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["train"][0]["clip_id"] == "fight:clip_1"
    assert payload["train"][0]["class_name"] == "fight"


def test_materialize_split_plan_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "fight" / "clip_1.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("bad", encoding="utf-8")
    plan = {
        "train": [ClipItem("fight:clip_1", "fight", str(source))],
        "val": [],
        "test": [],
    }

    with pytest.raises(ValueError, match="Unsupported clip file type"):
        dataset_split.materialize_split_plan(plan, tmp_path / "dataset")


def test_materialize_split_plan_rejects_invalid_split_keys(tmp_path: Path) -> None:
    plan = {
        "train": [],
        "valid": [],
        "test": [],
    }

    with pytest.raises(ValueError, match="train/val/test"):
        dataset_split.materialize_split_plan(plan, tmp_path / "dataset")


def test_extract_mp4_to_frames_writes_expected_number_of_frames(tmp_path: Path, monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self._frames = [object(), object()]

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, object | None]:
            if self._frames:
                return True, self._frames.pop(0)
            return False, None

        def release(self) -> None:
            return None

    def fake_imwrite(path: str, _frame: object) -> bool:
        Path(path).write_bytes(b"frame")
        return True

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda _path: FakeCapture(),
        imwrite=fake_imwrite,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    output_dir = tmp_path / "frames"
    dataset_split.extract_mp4_to_frames(tmp_path / "clip.mp4", output_dir)

    assert [path.name for path in sorted(output_dir.glob("*.jpg"))] == ["000001.jpg", "000002.jpg"]


def test_extract_mp4_to_frames_raises_for_unopenable_video(tmp_path: Path, monkeypatch) -> None:
    class FakeCapture:
        def isOpened(self) -> bool:
            return False

        def read(self) -> tuple[bool, object | None]:
            return False, None

        def release(self) -> None:
            return None

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda _path: FakeCapture(),
        imwrite=lambda _path, _frame: True,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    with pytest.raises(ValueError, match="Unable to open video clip"):
        dataset_split.extract_mp4_to_frames(tmp_path / "broken.mp4", tmp_path / "frames")
