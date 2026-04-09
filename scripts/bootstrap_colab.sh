#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="${1:-}"

if [[ -n "${INPUT_ROOT}" ]]; then
  SCRIPT_DIR="${INPUT_ROOT}/scripts"
  if [[ ! -f "${SCRIPT_DIR}/bootstrap_colab.sh" ]]; then
    echo "Unable to locate scripts/bootstrap_colab.sh under: ${INPUT_ROOT}" >&2
    exit 1
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${PKG_DIR}/.." && pwd)"

cd "${WORK_ROOT}"
python -m pip install --upgrade pip
# Colab often ships JAX packages that require NumPy 2.x, while this repo pins
# NumPy 1.26.x for the current training/export toolchain. Remove the conflicting
# preinstalls before resolving this environment.
CONFLICT_PACKAGES=()
if python -m pip show jax >/dev/null 2>&1; then
  CONFLICT_PACKAGES+=(jax)
fi
if python -m pip show jaxlib >/dev/null 2>&1; then
  CONFLICT_PACKAGES+=(jaxlib)
fi
if [[ ${#CONFLICT_PACKAGES[@]} -gt 0 ]]; then
  echo "Removing conflicting preinstalled packages: ${CONFLICT_PACKAGES[*]}"
  python -m pip uninstall -y "${CONFLICT_PACKAGES[@]}"
fi
python -m pip install -r "${PKG_DIR}/requirements-colab.txt"

mkdir -p /content/raw_dataset
mkdir -p /content/dataset
mkdir -p /content/checkpoints
mkdir -p /content/artifacts

echo "Colab bootstrap complete."
echo "Run from: ${WORK_ROOT}"
echo "Next: bash ${PKG_DIR}/scripts/colab_run_training.sh ${PKG_DIR}"
