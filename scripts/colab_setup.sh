#!/usr/bin/env bash
set -euo pipefail

# ── Usage ────────────────────────────────────────────────────────────
# bash collab_setup.sh <github_repo_url> [options]
#
# Options:
#   --branch <branch>             Git branch (default: main)
#   --repo-dir <path>             Clone target (default: /content/collab_training_anomaly_detection)
#   --raw-dataset-dir <path>      Raw clips location (default: /content/raw_dataset)
#   --dataset-dir <path>          Prepared splits (default: /content/dataset)
#   --checkpoint-dir <path>       Local checkpoints (default: /content/checkpoints)
#   --drive-checkpoint-dir <path> Drive backup dir (default: /content/drive/MyDrive/action_model_checkpoints)
#   --artifact-dir <path>         Export artifacts (default: /content/artifacts)
#   --kaggle-dataset <slug>       Kaggle dataset to pull (optional)
#   --kaggle-clean                Wipe raw-dataset-dir before pull
# ─────────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <github_repo_url> [options]" >&2
  exit 1
fi

REPO_URL="$1"; shift

BRANCH="main"
REPO_DIR="/content/collab_training_anomaly_detection"
RAW_DATASET_DIR="/content/raw_dataset"
DATASET_DIR="/content/dataset"
CHECKPOINT_DIR="/content/checkpoints"
DRIVE_CHECKPOINT_DIR="/content/drive/MyDrive/action_model_checkpoints"
ARTIFACT_DIR="/content/artifacts"
KAGGLE_DATASET=""
KAGGLE_CLEAN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)           BRANCH="$2";               shift 2 ;;
    --repo-dir)         REPO_DIR="$2";              shift 2 ;;
    --raw-dataset-dir)  RAW_DATASET_DIR="$2";       shift 2 ;;
    --dataset-dir)      DATASET_DIR="$2";           shift 2 ;;
    --checkpoint-dir)   CHECKPOINT_DIR="$2";        shift 2 ;;
    --drive-checkpoint-dir) DRIVE_CHECKPOINT_DIR="$2"; shift 2 ;;
    --artifact-dir)     ARTIFACT_DIR="$2";          shift 2 ;;
    --kaggle-dataset)   KAGGLE_DATASET="$2";        shift 2 ;;
    --kaggle-clean)     KAGGLE_CLEAN=true;          shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── 1. Clone ─────────────────────────────────────────────────────────
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "Cloning ${REPO_URL} (branch: ${BRANCH}) -> ${REPO_DIR}"
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
else
  echo "Repo exists at ${REPO_DIR}; pulling latest..."
  git -C "${REPO_DIR}" pull
fi

# ── 2. Bootstrap (install deps) ─────────────────────────────────────
if [[ -f "${REPO_DIR}/scripts/bootstrap_colab.sh" ]]; then
  BOOTSTRAP="${REPO_DIR}/scripts/bootstrap_colab.sh"
elif [[ -f "${REPO_DIR}/backend/collab_scripts/scripts/bootstrap_colab.sh" ]]; then
  BOOTSTRAP="${REPO_DIR}/backend/collab_scripts/scripts/bootstrap_colab.sh"
elif [[ -f "${REPO_DIR}/collab_scripts/scripts/bootstrap_colab.sh" ]]; then
  BOOTSTRAP="${REPO_DIR}/collab_scripts/scripts/bootstrap_colab.sh"
else
  echo "Could not find scripts/bootstrap_colab.sh" >&2; exit 1
fi
bash "${BOOTSTRAP}" "${REPO_DIR}"

# ── 3. Locate and update pipeline config ─────────────────────────────
if [[ -f "${REPO_DIR}/pipeline_config.json" ]]; then
  CONFIG_PATH="${REPO_DIR}/pipeline_config.json"
elif [[ -f "${REPO_DIR}/backend/collab_scripts/pipeline_config.json" ]]; then
  CONFIG_PATH="${REPO_DIR}/backend/collab_scripts/pipeline_config.json"
elif [[ -f "${REPO_DIR}/collab_scripts/pipeline_config.json" ]]; then
  CONFIG_PATH="${REPO_DIR}/collab_scripts/pipeline_config.json"
else
  echo "pipeline_config.json not found" >&2; exit 1
fi

python3 - "${CONFIG_PATH}" "${RAW_DATASET_DIR}" "${DATASET_DIR}" \
  "${CHECKPOINT_DIR}" "${DRIVE_CHECKPOINT_DIR}" "${ARTIFACT_DIR}" <<'PY'
import json, sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
config.setdefault("paths", {})
config["paths"]["raw_dataset_dir"]        = sys.argv[2]
config["paths"]["dataset_dir"]            = sys.argv[3]
config["paths"]["checkpoint_dir"]         = sys.argv[4]
config["paths"]["drive_checkpoint_dir"]   = sys.argv[5]
config["paths"]["artifact_dir"]           = sys.argv[6]
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(f"Config updated: {config_path}")
for k, v in config["paths"].items():
    print(f"  {k}: {v}")
PY

# ── 4. Optional Kaggle dataset pull ──────────────────────────────────
if [[ -n "${KAGGLE_DATASET}" ]]; then
  if [[ -f "${REPO_DIR}/scripts/colab_pull_kaggle_dataset.sh" ]]; then
    PULL_SCRIPT="${REPO_DIR}/scripts/colab_pull_kaggle_dataset.sh"
  elif [[ -f "${REPO_DIR}/backend/collab_scripts/scripts/colab_pull_kaggle_dataset.sh" ]]; then
    PULL_SCRIPT="${REPO_DIR}/backend/collab_scripts/scripts/colab_pull_kaggle_dataset.sh"
  elif [[ -f "${REPO_DIR}/collab_scripts/scripts/colab_pull_kaggle_dataset.sh" ]]; then
    PULL_SCRIPT="${REPO_DIR}/collab_scripts/scripts/colab_pull_kaggle_dataset.sh"
  else
    echo "colab_pull_kaggle_dataset.sh not found" >&2; exit 1
  fi

  PULL_CMD=(bash "${PULL_SCRIPT}" "${KAGGLE_DATASET}" --raw-dataset-dir "${RAW_DATASET_DIR}")
  if [[ "${KAGGLE_CLEAN}" == true ]]; then
    PULL_CMD+=(--clean)
  fi
  "${PULL_CMD[@]}"
fi

echo ""
echo "===== Setup complete ====="
echo "Repo:       ${REPO_DIR}"
echo "Config:     ${CONFIG_PATH}"
echo "Next: bash ${REPO_DIR}/scripts/colab_run_training.sh ${REPO_DIR}"
