#!/usr/bin/env bash
set -euo pipefail
ETA_LAB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ETA_LAB_ROOT"
ETA_VERSION=$(conda run -n pytorch python -c 'from pathlib import Path; import re; versions = [p for p in Path.cwd().glob("EtaZero_V*") if p.is_dir() and re.fullmatch(r"EtaZero_V\d+(?:\.\d+)*", p.name)]; print(max(versions, key=lambda p: tuple(map(int, p.name.removeprefix("EtaZero_V").split(".")))).name)')
bash "$ETA_VERSION/scripts/build.sh"
export PYTHONPATH="$ETA_LAB_ROOT/$ETA_VERSION/python:$ETA_LAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n pytorch python -m web.server "$@"
