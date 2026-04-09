from __future__ import annotations

from collab_scripts.artifacts import build_artifact_filenames


def test_build_artifact_filenames_uses_version_and_date() -> None:
    names = build_artifact_filenames(model_version="1.2.0", date_str="2026-04-09")

    assert names.pt == "action_model_v1.2.0_2026-04-09.pt"
    assert names.onnx == "action_model_v1.2.0_2026-04-09.onnx"
    assert names.xml == "action_model_v1.2.0_2026-04-09.xml"
    assert names.bin == "action_model_v1.2.0_2026-04-09.bin"
