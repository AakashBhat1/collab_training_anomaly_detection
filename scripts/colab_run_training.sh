#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/collab_scripts"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  REPO_ROOT="$1"
  shift
fi

CONFIG_PATH=""
AUTO_RESUME=true
KAGGLE_DATASET=""
RAW_DATASET_DIR=""
KAGGLE_CLEAN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --no-auto-resume)
      AUTO_RESUME=false
      shift
      ;;
    --kaggle-dataset)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --kaggle-dataset" >&2
        exit 1
      fi
      KAGGLE_DATASET="$2"
      shift 2
      ;;
    --raw-dataset-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --raw-dataset-dir" >&2
        exit 1
      fi
      RAW_DATASET_DIR="$2"
      shift 2
      ;;
    --kaggle-clean)
      KAGGLE_CLEAN=true
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Repo root IS the package — run python -m from parent dir
WORK_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/pipeline_config.json"

if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${DEFAULT_CONFIG}"
fi

cd "${WORK_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -n "${KAGGLE_DATASET}" ]]; then
  PULL_SCRIPT="${REPO_ROOT}/scripts/colab_pull_kaggle_dataset.sh"
  if [[ ! -f "${PULL_SCRIPT}" ]]; then
    echo "Kaggle pull script not found: ${PULL_SCRIPT}" >&2
    exit 1
  fi

  if [[ -z "${RAW_DATASET_DIR}" ]]; then
    RAW_DATASET_DIR="$(python - "${CONFIG_PATH}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding='utf-8'))
raw_dir = str(config.get('paths', {}).get('raw_dataset_dir', '/content/raw_dataset'))
print(raw_dir)
PY
)"
  fi

  PULL_COMMAND=(bash "${PULL_SCRIPT}" "${KAGGLE_DATASET}" "${RAW_DATASET_DIR}")
  if [[ "${KAGGLE_CLEAN}" == true ]]; then
    PULL_COMMAND+=(--clean)
  fi

  echo "Pulling dataset from Kaggle: ${KAGGLE_DATASET}"
  "${PULL_COMMAND[@]}"
fi

echo "Running pipeline from ${WORK_ROOT} with config ${CONFIG_PATH}"
COMMAND=(python -m "${PACKAGE_NAME}.run_pipeline" --config "${CONFIG_PATH}")
if [[ "${AUTO_RESUME}" == true ]]; then
  COMMAND+=(--auto-resume)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  COMMAND+=("${EXTRA_ARGS[@]}")
fi

"${COMMAND[@]}"
