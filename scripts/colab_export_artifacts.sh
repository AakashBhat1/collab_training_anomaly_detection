#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/content/intruder_detection_system}"
OUTPUT_ARCHIVE="${2:-/content/action_model_export.tgz}"

if [[ -d "${REPO_ROOT}/backend/collab_scripts" ]]; then
  WORK_ROOT="${REPO_ROOT}/backend"
  CONFIG_PATH="${WORK_ROOT}/collab_scripts/pipeline_config.json"
elif [[ -d "${REPO_ROOT}/collab_scripts" ]]; then
  WORK_ROOT="${REPO_ROOT}"
  CONFIG_PATH="${WORK_ROOT}/collab_scripts/pipeline_config.json"
else
  echo "Could not find collab_scripts in ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mapfile -t PATH_VALUES < <(python - "${CONFIG_PATH}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
paths = config.get("paths", {})


def resolve(raw_value: str, default_value: str) -> str:
  value = str(raw_value or default_value)
  path_obj = Path(value)
  if not path_obj.is_absolute():
    path_obj = (config_path.parent / path_obj).resolve()
  return str(path_obj)


print(resolve(paths.get("artifact_dir"), "/content/artifacts"))
print(resolve(paths.get("checkpoint_dir"), "/content/checkpoints"))
print(resolve(paths.get("dataset_dir"), "/content/dataset"))
print(config.get("model_version", "0.1.0"))
PY
)

ARTIFACT_DIR="${PATH_VALUES[0]}"
CHECKPOINT_DIR="${PATH_VALUES[1]}"
DATASET_DIR="${PATH_VALUES[2]}"
MODEL_VERSION="${PATH_VALUES[3]}"

STAGING_DIR="/tmp/action_model_export_${MODEL_VERSION}"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

if [[ ! -d "${ARTIFACT_DIR}" ]]; then
  echo "artifact_dir not found: ${ARTIFACT_DIR}" >&2
  exit 1
fi

if ! find "${ARTIFACT_DIR}" -maxdepth 1 -type f \( -name "*.xml" -o -name "*.onnx" -o -name "*.bin" \) | grep -q .; then
  echo "No model export files (.xml/.bin/.onnx) found in artifact_dir: ${ARTIFACT_DIR}" >&2
  exit 1
fi

cp -r "${ARTIFACT_DIR}" "${STAGING_DIR}/artifacts"

if [[ -d "${CHECKPOINT_DIR}" ]]; then
  mkdir -p "${STAGING_DIR}/checkpoints"
  for file_name in last.pt best.pt training_summary.json; do
    if [[ -f "${CHECKPOINT_DIR}/${file_name}" ]]; then
      cp "${CHECKPOINT_DIR}/${file_name}" "${STAGING_DIR}/checkpoints/${file_name}"
    fi
  done
fi

if [[ -f "${DATASET_DIR}/split_manifest.json" ]]; then
  mkdir -p "${STAGING_DIR}/dataset"
  cp "${DATASET_DIR}/split_manifest.json" "${STAGING_DIR}/dataset/split_manifest.json"
fi

cat > "${STAGING_DIR}/EXPORT_INFO.txt" <<EOF
model_version=${MODEL_VERSION}
artifact_dir=${ARTIFACT_DIR}
checkpoint_dir=${CHECKPOINT_DIR}
dataset_dir=${DATASET_DIR}
created_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

mkdir -p "$(dirname "${OUTPUT_ARCHIVE}")"
tar -czf "${OUTPUT_ARCHIVE}" -C "${STAGING_DIR}" .

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${OUTPUT_ARCHIVE}" > "${OUTPUT_ARCHIVE}.sha256"
  echo "SHA256 written to ${OUTPUT_ARCHIVE}.sha256"
fi

echo "Export archive created: ${OUTPUT_ARCHIVE}"
