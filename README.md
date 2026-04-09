# Colab Training Handoff (MVP)

This folder is the GitHub-uploadable training/export bundle for the MVP action classifier:
- classes: `fight`, `theft`, `intrusion`, `normal`
- pipeline: split -> train -> evaluate -> export ONNX/OpenVINO
- target runtime artifact: OpenVINO IR (`.xml` + `.bin`)

## 1) Expected Data Layout

Raw clips under `/content/raw_dataset`:

```text
/content/raw_dataset/
  fight/
    clip_001/               # preferred: folder of extracted frames
      0001.jpg
      ...
    clip_002.mp4            # supported: converted to frames during prepare step
  theft/
  intrusion/
  normal/
```

Notes:
- During `prepare_dataset`, every source clip is materialized as a frame directory under `train|val|test/<class>/<clip_id>/`.
- Re-running `prepare_dataset` rebuilds split folders from scratch to prevent stale data leakage.

Prepared split output (generated automatically):

```text
/content/dataset/
  train/<class>/...
  val/<class>/...
  test/<class>/...
  split_manifest.json
```

## 2) Colab Setup

From Colab:

```bash
git clone <your-repo-url> /content/collab_scripts
cd /content/collab_scripts
bash scripts/bootstrap_colab.sh /content/collab_scripts
```

GitHub-first helper scripts (recommended):

```bash
# 1) clone + bootstrap from GitHub
bash scripts/colab_clone_and_bootstrap.sh \
  https://github.com/<org>/<repo>.git \
  /content/collab_scripts \
  main

# 2) run split/train/eval/export with auto-resume
bash scripts/colab_run_training.sh /content/collab_scripts

# optional: pull dataset from Kaggle before training
bash scripts/colab_run_training.sh \
  /content/collab_scripts \
  --config collab_scripts/pipeline_config.json \
  --kaggle-dataset <owner>/<dataset-slug> \
  --kaggle-clean

# with explicit config and passthrough pipeline flags
bash scripts/colab_run_training.sh \
  /content/collab_scripts \
  --config collab_scripts/pipeline_config.json \
  --skip-export

# 3) package outputs for download/handoff
bash scripts/colab_export_artifacts.sh \
  /content/collab_scripts \
  /content/action_model_export.tgz
```

Helper scripts added in this package:
- `scripts/colab_clone_and_bootstrap.sh`
- `scripts/colab_run_training.sh`
- `scripts/colab_export_artifacts.sh`
- `scripts/colab_pull_kaggle_dataset.sh`

Kaggle credentials:
- Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables in Colab, or
- place `kaggle.json` at `~/.kaggle/kaggle.json` with correct permissions.

KaggleHub support:
- `requirements-colab.txt` now includes `kagglehub[pandas-datasets]`.
- The automation notebook includes optional `kagglehub.load_dataset(...)` support using your dataset slug and file path.
- In `notebooks/colab_training_automation.ipynb`, set:
  - `USE_KAGGLEHUB_PANDAS_PREVIEW = True`
  - `KAGGLEHUB_FILE_PATH = "<path-inside-dataset>"`
  - optional `KAGGLEHUB_SQL_QUERY`

If using Drive-backed checkpoint resume:

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3) Configure Paths/Hyperparams

Edit:
- `collab_scripts/pipeline_config.json`

Important fields:
- `paths.raw_dataset_dir`
- `paths.dataset_dir`
- `paths.checkpoint_dir`
- `paths.drive_checkpoint_dir`
- `paths.artifact_dir`
- `model_version`

## 4) Run Full Pipeline

Run from the parent of the `collab_scripts` package:
- backend layout: `/content/collab_scripts/backend`
- root layout: `/content/collab_scripts`

```bash
python -m collab_scripts.run_pipeline --auto-resume
```

Step-by-step alternative:

```bash
python -m collab_scripts.prepare_dataset --config collab_scripts/pipeline_config.json
python -m collab_scripts.train_action_model --config collab_scripts/pipeline_config.json --auto-resume
python -m collab_scripts.evaluate_action_model --config collab_scripts/pipeline_config.json
python -m collab_scripts.export_openvino --config collab_scripts/pipeline_config.json
```

## 5) Outputs

- checkpoints:
  - `/content/checkpoints/last.pt`
  - `/content/checkpoints/best.pt`
- evaluation:
  - `/content/artifacts/evaluation_report.json`
- model artifacts:
  - `/content/artifacts/action_model_v<version>_<date>.onnx`
  - `/content/artifacts/action_model_v<version>_<date>.xml`
  - `/content/artifacts/action_model_v<version>_<date>.bin`
- packaged export bundle (if using helper script):
  - `/content/action_model_export.tgz`

## 6) Verification Commands (TDD Loop)

From the package parent directory:

```bash
python -m pytest collab_scripts/tests -k split -q
python -m pytest collab_scripts/tests -q
python -m pytest collab_scripts/tests --cov=collab_scripts --cov-report=term-missing
```

## 7) Backend Handoff

Copy exported OpenVINO files into:

`backend/yolo_classifier/models/`

Use the same version string in backend runtime config so alerts and metadata remain traceable.

