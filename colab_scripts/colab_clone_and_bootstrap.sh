#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <github_repo_url> [target_dir] [branch]" >&2
  echo "Example: $0 https://github.com/org/repo.git /content/collab_training_anomaly_detection main" >&2
  exit 1
fi

REPO_URL="$1"
TARGET_DIR="${2:-/content/collab_training_anomaly_detection}"
BRANCH="${3:-}"

if [[ -d "${TARGET_DIR}" && ! -d "${TARGET_DIR}/.git" ]]; then
  if [[ -z "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]]; then
    rmdir "${TARGET_DIR}"
  else
    echo "Target directory exists but is not a git repository: ${TARGET_DIR}" >&2
    echo "Use a clean target directory or remove existing files first." >&2
    exit 1
  fi
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  if [[ -n "${BRANCH}" ]]; then
    git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${TARGET_DIR}"
  else
    git clone "${REPO_URL}" "${TARGET_DIR}"
  fi
else
  echo "Repository already exists at ${TARGET_DIR}; skipping clone."
fi

if [[ -f "${TARGET_DIR}/backend/collab_scripts/scripts/bootstrap_colab.sh" ]]; then
  BOOTSTRAP_SCRIPT="${TARGET_DIR}/backend/collab_scripts/scripts/bootstrap_colab.sh"
elif [[ -f "${TARGET_DIR}/collab_scripts/scripts/bootstrap_colab.sh" ]]; then
  BOOTSTRAP_SCRIPT="${TARGET_DIR}/collab_scripts/scripts/bootstrap_colab.sh"
else
  echo "Could not find scripts/bootstrap_colab.sh under ${TARGET_DIR}" >&2
  exit 1
fi

bash "${BOOTSTRAP_SCRIPT}" "${TARGET_DIR}"

echo "GitHub clone + Colab bootstrap complete."
