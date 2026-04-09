from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset prep, training, evaluation, and export.")
    parser.add_argument(
        "--config",
        default="collab_scripts/pipeline_config.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset split/materialization step.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip held-out test evaluation.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX/OpenVINO export.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Pass auto-resume into the training step.",
    )
    return parser.parse_args()


def _run(module: str, config_path: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, "-m", module, "--config", config_path]
    if extra_args:
        command.extend(extra_args)
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_prepare:
        _run("collab_scripts.prepare_dataset", args.config)

    train_args: list[str] = []
    if args.auto_resume:
        train_args.append("--auto-resume")
    _run("collab_scripts.train_action_model", args.config, train_args)

    if not args.skip_eval:
        _run("collab_scripts.evaluate_action_model", args.config)
    if not args.skip_export:
        _run("collab_scripts.export_openvino", args.config)


if __name__ == "__main__":
    main()

