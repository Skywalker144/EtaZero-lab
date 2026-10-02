#!/usr/bin/env bash
set -euo pipefail
ETA_LAB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ETA_LAB_ROOT"
bash EtaZero_V0/scripts/build.sh
export PYTHONPATH="$ETA_LAB_ROOT/EtaZero_V0/python:$ETA_LAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n pytorch python -m web.server "$@"
