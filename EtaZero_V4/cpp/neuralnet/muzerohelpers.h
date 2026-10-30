#pragma once
#include <cuda_runtime.h>

// Learned layers use the source backend. These kernels only normalize/pack the
// MuZero state boundary. State is always FP32 NCHW, with its mask last.
void muZeroNormalize(const void* values, const float* mask, float* output,
                     int batch, int channels, int area, bool fp16, bool nhwc, cudaStream_t stream);
void muZeroPack(const float* const* states, const float* contiguous, const int* actions, float* output,
                int batch, int channels, int area, int actionPlanes, bool nhwc, cudaStream_t stream);
