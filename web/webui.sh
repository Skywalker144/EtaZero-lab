#!/usr/bin/env bash
set -euo pipefail
ETA_LAB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ETA_LAB_ROOT"
if [[ -z "${CONDA_PREFIX:-}" || "$(basename -- "$CONDA_PREFIX")" != pytorch ]]; then
    exec conda run --no-capture-output -n pytorch bash "$ETA_LAB_ROOT/web/webui.sh" "$@"
fi
export PYTHONPATH="$ETA_LAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m web.launch "$@"
