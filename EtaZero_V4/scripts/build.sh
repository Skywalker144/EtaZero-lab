#!/usr/bin/env bash
set -euo pipefail
ETA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${CONDA_PREFIX:-}" || "$(basename -- "$CONDA_PREFIX")" != pytorch ]]; then
    exec conda run --no-capture-output -n pytorch bash "$ETA_ROOT/scripts/build.sh" "$@"
fi
ETA_CUDNN_ROOT=$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')
ETA_CUBLAS_ROOT=$(python -c 'import nvidia.cu13; print(nvidia.cu13.__path__[0])')
cmake -S "$ETA_ROOT/cpp" -B "$ETA_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release -DUSE_BACKEND=CUDA -DBUILD_DISTRIBUTED=OFF \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
    -DCUDNN_INCLUDE_DIR="$ETA_CUDNN_ROOT/include" -DCUDNN_LIBRARY="$ETA_CUDNN_ROOT/lib/libcudnn.so.9" \
    -DCUDA_cublas_LIBRARY="$ETA_CUBLAS_ROOT/lib/libcublas.so.13" \
    -DCUDA_cublasLt_LIBRARY="$ETA_CUBLAS_ROOT/lib/libcublasLt.so.13" \
    -DETAZERO_CUBLAS_INCLUDE_DIR="$ETA_CUBLAS_ROOT/include" \
    -DETAZERO_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-120}" \
    "$@"
cmake --build "$ETA_ROOT/build" --parallel "${BUILD_JOBS:-2}"
