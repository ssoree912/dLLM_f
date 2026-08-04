#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_ROOT:?Set INPUT_ROOT to a schema-v3 teacher shard directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
export PYTHONPATH="${EXPERIMENT_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --input-root "${INPUT_ROOT}" \
  --budget "${BUDGET:-128}"
)
if [[ -n "${OUTPUT_JSON:-}" ]]; then
  ARGS+=(--output "${OUTPUT_JSON}")
fi

"${PYTHON_BIN:-python}" -m step_distill.oracle_diagnostics "${ARGS[@]}"
