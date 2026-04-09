#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <kaggle_dataset_slug> [--raw-dataset-dir <path>] [--clean]" >&2
  echo "Example: $0 owner/dataset-name --raw-dataset-dir /content/raw_dataset --clean" >&2
  exit 1
fi

KAGGLE_DATASET_SLUG="$1"
shift

RAW_DATASET_DIR="/content/raw_dataset"
RAW_DATASET_DIR_SET=false
CLEAN_TARGET=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-dataset-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --raw-dataset-dir" >&2
        exit 1
      fi
      RAW_DATASET_DIR="$2"
      RAW_DATASET_DIR_SET=true
      shift 2
      ;;
    --clean)
      CLEAN_TARGET=true
      shift
      ;;
    *)
      # Backward compatible positional raw dataset dir.
      if [[ "${RAW_DATASET_DIR_SET}" == false && "$1" != -* ]]; then
        RAW_DATASET_DIR="$1"
        RAW_DATASET_DIR_SET=true
        shift
      else
        echo "Unknown argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

RAW_DATASET_DIR="$(python - "${RAW_DATASET_DIR}" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1]).expanduser().resolve(strict=False)
print(str(target))
PY
)"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install requirements first (bootstrap_colab.sh)." >&2
  exit 1
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "Kaggle credentials missing. Set KAGGLE_USERNAME/KAGGLE_KEY or provide ~/.kaggle/kaggle.json." >&2
    exit 1
  fi
fi

TMP_DIR="/tmp/kaggle_dataset_download"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

if [[ "${CLEAN_TARGET}" == true ]]; then
  if [[ "${RAW_DATASET_DIR}" == /content/drive/* ]]; then
    echo "Refusing to clean Google Drive path: ${RAW_DATASET_DIR}" >&2
    exit 1
  fi
  case "${RAW_DATASET_DIR}" in
    ""|"/"|"/content"|"/content/"|"/content/drive"|"/content/drive/"|"/content/drive/MyDrive"|"/content/drive/MyDrive/")
      echo "Refusing to clean unsafe path: ${RAW_DATASET_DIR}" >&2
      exit 1
      ;;
  esac
  if [[ "${RAW_DATASET_DIR}" != /content/* ]]; then
    echo "Refusing to clean path outside /content: ${RAW_DATASET_DIR}" >&2
    exit 1
  fi
  rm -rf "${RAW_DATASET_DIR}"
fi
mkdir -p "${RAW_DATASET_DIR}"

echo "Downloading Kaggle dataset: ${KAGGLE_DATASET_SLUG}"
kaggle datasets download -d "${KAGGLE_DATASET_SLUG}" -p "${TMP_DIR}" --unzip --force

# Move extracted content into RAW_DATASET_DIR. If archive has one top-level folder, flatten it.
shopt -s dotglob nullglob
entries=("${TMP_DIR}"/*)
if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
  cp -R "${entries[0]}"/* "${RAW_DATASET_DIR}"/
else
  for item in "${TMP_DIR}"/*; do
    cp -R "${item}" "${RAW_DATASET_DIR}"/
  done
fi
shopt -u dotglob nullglob

echo "Kaggle dataset ready at: ${RAW_DATASET_DIR}"
find "${RAW_DATASET_DIR}" -maxdepth 2 -type d | head -n 30
