#!/usr/bin/env bash
set -euo pipefail
ETA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ETA_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n pytorch python -m etazero "$@"
