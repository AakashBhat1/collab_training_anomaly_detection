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

TMP_DIR="/content/.kaggle_download"
ZIP_DIR="${TMP_DIR}/zip"
rm -rf "${TMP_DIR}"
mkdir -p "${ZIP_DIR}"

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
kaggle datasets download -d "${KAGGLE_DATASET_SLUG}" -p "${ZIP_DIR}" --force

shopt -s nullglob
zip_files=("${ZIP_DIR}"/*.zip)
shopt -u nullglob
if [[ ${#zip_files[@]} -ne 1 ]]; then
  echo "Expected one downloaded zip in ${ZIP_DIR}, found ${#zip_files[@]}" >&2
  exit 1
fi
ZIP_PATH="${zip_files[0]}"

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip command not found in runtime." >&2
  exit 1
fi

TOP_LEVEL_DIR="$(python - "${ZIP_PATH}" <<'PY'
import sys
import zipfile
from pathlib import PurePosixPath

zip_path = sys.argv[1]
top_levels = set()
with zipfile.ZipFile(zip_path) as zf:
    for name in zf.namelist():
        parts = PurePosixPath(name).parts
        if not parts:
            continue
        first = parts[0]
        if first in ("", "__MACOSX"):
            continue
        top_levels.add(first)

if len(top_levels) == 1:
    print(next(iter(top_levels)))
PY
)"

echo "Extracting dataset archive to disk: ${ZIP_PATH}"
unzip -oq "${ZIP_PATH}" -d "${RAW_DATASET_DIR}"
rm -f "${ZIP_PATH}"

# If the archive is wrapped in a single top-level folder, flatten it in place.
if [[ -n "${TOP_LEVEL_DIR}" && -d "${RAW_DATASET_DIR}/${TOP_LEVEL_DIR}" ]]; then
  shopt -s dotglob nullglob
  wrapped_entries=("${RAW_DATASET_DIR}/${TOP_LEVEL_DIR}"/*)
  if [[ ${#wrapped_entries[@]} -gt 0 ]]; then
    mv "${RAW_DATASET_DIR}/${TOP_LEVEL_DIR}"/* "${RAW_DATASET_DIR}/"
  fi
  shopt -u dotglob nullglob
  rmdir "${RAW_DATASET_DIR}/${TOP_LEVEL_DIR}" 2>/dev/null || true
fi

rm -rf "${TMP_DIR}"

echo "Kaggle dataset ready at: ${RAW_DATASET_DIR}"
find "${RAW_DATASET_DIR}" -maxdepth 2 -type d | head -n 30
