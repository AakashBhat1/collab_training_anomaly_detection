from __future__ import annotations

import argparse
from pathlib import Path

from collab_scripts.config_schema import load_pipeline_config
from collab_scripts.dataset_split import (
    discover_clips,
    materialize_split_plan,
    plan_split,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/val/test dataset splits.")
    parser.add_argument(
        "--config",
        default="collab_scripts/pipeline_config.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for split planning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    if config.paths is None:
        raise ValueError("`paths` must be defined in pipeline config for dataset preparation.")

    raw_dir = Path(config.paths.raw_dataset_dir)
    output_dir = Path(config.paths.dataset_dir)

    clips = discover_clips(source_dir=raw_dir, classes=config.classes)
    if not clips:
        raise ValueError(f"No clips discovered under {raw_dir}")

    split_plan = plan_split(items=clips, splits=config.splits, seed=args.seed)
    materialize_split_plan(plan=split_plan, output_dir=output_dir)
    write_split_manifest(output_dir / "split_manifest.json", split_plan)

    print(f"Prepared dataset at: {output_dir}")
    for split_name, split_items in split_plan.items():
        print(f"  {split_name}: {len(split_items)} clips")


if __name__ == "__main__":
    main()

