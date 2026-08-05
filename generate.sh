#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

OUTPUT="${OUTPUT:-${ROOT_DIR}/generated/input.json}"
REPORT="${REPORT:-${ROOT_DIR}/generated/input.report.json}"
STRATEGY="${STRATEGY:-${ROOT_DIR}/strategies/release_template_strategy.json}"

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${REPORT}")"

python -m crystal_llm.generate \
  --target-count "${TARGET_COUNT:-1000}" \
  --candidate-multiplier "${CANDIDATE_MULTIPLIER:-10}" \
  --max-per-formula "${MAX_PER_FORMULA:-1}" \
  --seed "${SEED:-20260502}" \
  --jobs "${JOBS:-1}" \
  --strategy "${STRATEGY}" \
  --training-data "${TRAINING_DATA:-}" \
  --output "${OUTPUT}" \
  --report "${REPORT}" \
  "$@"

python -m crystal_llm.validate_output --input "${OUTPUT}"
echo "Generated: ${OUTPUT}"
echo "Report: ${REPORT}"

