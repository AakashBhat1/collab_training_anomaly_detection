#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="${1:-}"

if [[ -n "${INPUT_ROOT}" ]]; then
	if [[ -f "${INPUT_ROOT}/backend/collab_scripts/bootstrap_colab.sh" ]]; then
		SCRIPT_DIR="${INPUT_ROOT}/backend/collab_scripts"
	elif [[ -f "${INPUT_ROOT}/collab_scripts/bootstrap_colab.sh" ]]; then
		SCRIPT_DIR="${INPUT_ROOT}/collab_scripts"
	elif [[ -f "${INPUT_ROOT}/bootstrap_colab.sh" ]]; then
		SCRIPT_DIR="${INPUT_ROOT}"
	else
		echo "Unable to locate collab_scripts under: ${INPUT_ROOT}" >&2
		exit 1
	fi
else
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WORK_ROOT}"
python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements-colab.txt"

mkdir -p /content/raw_dataset
mkdir -p /content/dataset
mkdir -p /content/checkpoints
mkdir -p /content/artifacts

echo "Colab bootstrap complete."
echo "Run from: ${WORK_ROOT}"
echo "Next: python -m collab_scripts.run_pipeline --auto-resume"

