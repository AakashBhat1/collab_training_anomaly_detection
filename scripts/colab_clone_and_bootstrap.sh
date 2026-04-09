#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <github_repo_url> [target_dir] [branch]" >&2
  echo "Example: $0 https://github.com/org/repo.git /content/collab_scripts main" >&2
  exit 1
fi

REPO_URL="$1"
TARGET_DIR="${2:-/content/collab_scripts}"
BRANCH="${3:-}"

if [[ "$(basename "${TARGET_DIR}")" != "collab_scripts" ]]; then
  echo "Target directory must end with /collab_scripts: ${TARGET_DIR}" >&2
  exit 1
fi

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

BOOTSTRAP_SCRIPT="${TARGET_DIR}/scripts/bootstrap_colab.sh"
if [[ ! -f "${BOOTSTRAP_SCRIPT}" ]]; then
  echo "Could not find bootstrap_colab.sh at ${BOOTSTRAP_SCRIPT}" >&2
  exit 1
fi

bash "${BOOTSTRAP_SCRIPT}" "${TARGET_DIR}"

echo "GitHub clone + Colab bootstrap complete."