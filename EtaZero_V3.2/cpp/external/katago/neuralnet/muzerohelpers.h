#pragma once
#include <cuda_runtime.h>
// EtaZero's latent-only extension; source kernels handle the learned network.
void etaNormalizeLatent(const void* values,const float* mask,float* output,int batch,int channels,int area,bool half,bool nhwc,cudaStream_t stream);
void etaPackLatent(const float* const* latents,const int* source,const int* target,float* output,int batch,int channels,int area,int actionPlanes,cudaStream_t stream);
