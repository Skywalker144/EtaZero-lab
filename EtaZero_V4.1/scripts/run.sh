#!/usr/bin/env bash
set -euo pipefail
ETA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${CONDA_PREFIX:-}" || "$(basename -- "$CONDA_PREFIX")" != pytorch ]]; then
    # Keep the controller as the foreground process so Ctrl+C can wait for its
    # children to drain instead of returning through an interrupted conda run.
    set +u  # Conda package activation hooks may read unset environment variables.
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate pytorch
    set -u
fi
export PYTHONPATH="$ETA_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -eq 0 || ${1:-} == --* && ${1:-} != --help && ${1:-} != --version ]]; then
    set -- run "$@"
fi
exec python -m etazero "$@"
