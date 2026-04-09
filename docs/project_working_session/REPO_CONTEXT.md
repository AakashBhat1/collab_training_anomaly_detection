# Repo Context

## Project Name
- collab_scripts: Colab-first training/export bundle for the MVP action classifier.

## Architecture Overview
- Config and contract:
  - `config_schema.py` validates pipeline JSON.
  - `artifacts.py` defines versioned artifact naming.
- Data preparation:
  - `dataset_split.py` discovers clips, plans deterministic train/val/test splits, materializes frame directories, and writes `split_manifest.json`.
  - `prepare_dataset.py` orchestrates split + materialization from config.
- Training/eval/export:
  - `training_data.py` loads frame-sequence clips for model input.
  - `model.py` defines the CNN+LSTM classifier.
  - `train_action_model.py` trains and checkpoints (`last.pt`, `best.pt`) with optional auto-resume.
  - `evaluate_action_model.py` writes held-out metrics (`evaluation_report.json`).
  - `export_openvino.py` exports ONNX then OpenVINO IR (`.xml` + `.bin`).
  - `run_pipeline.py` chains prepare/train/eval/export.
- Colab automation shell wrappers:
  - `bootstrap_colab.sh` installs dependencies and prepares standard Colab paths.
  - `colab_clone_and_bootstrap.sh` clones from GitHub and runs bootstrap in one command.
  - `colab_run_training.sh` runs full training pipeline with auto-resume defaults and optional automatic Kaggle dataset pull.
  - `colab_export_artifacts.sh` packages artifacts/checkpoints/split manifest into a portable archive.
  - `colab_pull_kaggle_dataset.sh` downloads and prepares raw dataset files from a Kaggle dataset slug.
- Colab automation notebook:
  - `colab_training_automation.ipynb` provides an end-to-end notebook flow for Drive mount, GitHub clone/bootstrap, optional Kaggle/KaggleHub dataset pull/preview, config path updates, training run, Drive backup checks, and resume-status checks.

## Key Dependencies
| Package | Purpose |
|---------|---------|
| torch/torchvision/torchaudio | training model and tensor ops |
| scikit-learn | recall/report/confusion metrics |
| pillow | image loading |
| opencv-python-headless | mp4-to-frame extraction during dataset prep |
| kaggle/kagglehub | Kaggle dataset download and optional pandas preview in notebook |
| onnx/openvino/openvino-dev | export and IR conversion |
| pytest/pytest-cov | contract tests and coverage checks |

## Environment
- Language/runtime: Python 3.13 in local venv, Colab Python for training runs.
- Package manager: pip.
- Test command:
  - Focused: `python -m pytest collab_scripts/tests -k split -q`
  - Full: `python -m pytest collab_scripts/tests -q`
  - Coverage: `python -m pytest collab_scripts/tests --cov=collab_scripts --cov-report=term-missing -q`

## Assumptions And Constraints
- MVP training classes are `fight`, `theft`, `intrusion`, `normal`.
- Intrusion alert authority remains ROI/rule logic in backend runtime; this package is training/export only.
- Dataset preparation now always materializes to frame directories and rebuilds split dirs from scratch to avoid stale data leakage.
- This pass does not include backend inference/runtime integration.
