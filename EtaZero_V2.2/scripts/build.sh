#!/usr/bin/env bash
set -euo pipefail
ETA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${CONDA_PREFIX:-}" || "$(basename -- "$CONDA_PREFIX")" != pytorch ]]; then
    exec conda run --no-capture-output -n pytorch bash "$ETA_ROOT/scripts/build.sh" "$@"
fi
ETA_TORCH_PREFIX=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S "$ETA_ROOT/cpp" -B "$ETA_ROOT/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ETA_TORCH_PREFIX;$CONDA_PREFIX" \
    -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" -DCUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX" \
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" "$@"
cmake --build "$ETA_ROOT/build" --parallel "${BUILD_JOBS:-2}"
python "$ETA_ROOT/scripts/write_build_manifest.py" "$ETA_ROOT/build"
