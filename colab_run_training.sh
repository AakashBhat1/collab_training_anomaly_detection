#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/content/intruder_detection_system"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  REPO_ROOT="$1"
  shift
fi

CONFIG_PATH=""
AUTO_RESUME=true
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
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -d "${REPO_ROOT}/backend/collab_scripts" ]]; then
  WORK_ROOT="${REPO_ROOT}/backend"
  DEFAULT_CONFIG="collab_scripts/pipeline_config.json"
elif [[ -d "${REPO_ROOT}/collab_scripts" ]]; then
  WORK_ROOT="${REPO_ROOT}"
  DEFAULT_CONFIG="collab_scripts/pipeline_config.json"
else
  echo "Could not find collab_scripts in ${REPO_ROOT}" >&2
  exit 1
fi

if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${DEFAULT_CONFIG}"
fi

cd "${WORK_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found from ${WORK_ROOT}: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "Running pipeline from ${WORK_ROOT} with config ${CONFIG_PATH}"
COMMAND=(python -m collab_scripts.run_pipeline --config "${CONFIG_PATH}")
if [[ "${AUTO_RESUME}" == true ]]; then
  COMMAND+=(--auto-resume)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  COMMAND+=("${EXTRA_ARGS[@]}")
fi

"${COMMAND[@]}"
