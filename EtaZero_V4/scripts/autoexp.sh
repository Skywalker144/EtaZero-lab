#!/usr/bin/env bash
set -euo pipefail
ETA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ETA_ROOT/scripts/run.sh" experiment "$@"
