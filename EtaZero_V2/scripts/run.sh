#!/usr/bin/env bash
set -euo pipefail
ETA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ETA_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -eq 0 || ${1:-} == --* && ${1:-} != --help ]]; then
    set -- run "$@"
fi
if [[ -n "${CONDA_PREFIX:-}" && "$(basename -- "$CONDA_PREFIX")" == pytorch ]]; then
    exec python -m etazero "$@"
fi
exec conda run --no-capture-output -n pytorch python -m etazero "$@"
