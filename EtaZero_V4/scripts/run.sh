#!/usr/bin/env bash
set -euo pipefail
ETA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${CONDA_PREFIX:-}" || "$(basename -- "$CONDA_PREFIX")" != pytorch ]]; then
    exec conda run --no-capture-output -n pytorch bash "$ETA_ROOT/scripts/run.sh" "$@"
fi
export PYTHONPATH="$ETA_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -eq 0 || ${1:-} == --* && ${1:-} != --help && ${1:-} != --version ]]; then
    set -- run "$@"
fi
exec python -m etazero "$@"
