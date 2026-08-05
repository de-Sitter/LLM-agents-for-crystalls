#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATOR_DIR="${EVALUATOR_DIR:-${ROOT_DIR}/external/evaluator}"
INPUT="${INPUT:-${ROOT_DIR}/input.json}"
TRAINING_DATA="${TRAINING_DATA:-${EVALUATOR_DIR}/data/a_training.json}"
PPD_PATH="${PPD_PATH:-${EVALUATOR_DIR}/data/2024-08-07-ppd-mp.pkl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/results.local.json}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -f "${EVALUATOR_DIR}/independent_evaluator.py" ]]; then
  echo "Evaluator not found: ${EVALUATOR_DIR}/independent_evaluator.py" >&2
  echo "Set EVALUATOR_DIR to a legal local copy; see docs/REPRODUCIBILITY.md." >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python "${EVALUATOR_DIR}/independent_evaluator.py" \
  --input "${INPUT}" \
  --training-data "${TRAINING_DATA}" \
  --output "${OUTPUT}" \
  --ppd-path "${PPD_PATH}" \
  --device "${DEVICE}" \
  "$@"

