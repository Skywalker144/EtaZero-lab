// Masked min/max adapted from EtaZero V3.4's MiniZero normalization; attribution
// is recorded in reference_sources.json and licenses/MiniZero.txt.
#include "muzerohelpers.h"
#include <cuda_fp16.h>
#include <cfloat>

namespace {
template<class T>
__global__ void normalize(const T* values, const float* mask, float* output, int channels, int area, bool nhwc) {
  __shared__ float minimum[256], maximum[256];
  const int row = blockIdx.x, lane = threadIdx.x;
  float lo = FLT_MAX, hi = -FLT_MAX;
  for(int j = lane; j < channels*area; j += blockDim.x) {
    const int c = j/area, point = j%area;
    if(mask == nullptr || mask[row*area+point] != 0) {
      const float v = float(values[row*channels*area + (nhwc ? point*channels+c : j)]);
      if(isnan(v)) lo = hi = v;
      else if(!isnan(lo)) { lo = fminf(lo,v); hi = fmaxf(hi,v); }
    }
  }
  minimum[lane] = lo; maximum[lane] = hi;
  __syncthreads();
  for(int stride = 128; stride > 0; stride >>= 1) {
    if(lane < stride) {
      if(isnan(minimum[lane+stride])) minimum[lane] = maximum[lane] = minimum[lane+stride];
      else if(!isnan(minimum[lane])) {
        minimum[lane] = fminf(minimum[lane],minimum[lane+stride]);
        maximum[lane] = fmaxf(maximum[lane],maximum[lane+stride]);
      }
    }
    __syncthreads();
  }
  const float span = maximum[0]-minimum[0];
  const float denominator = span < 1e-5f ? span+1e-5f : span;
  for(int j = lane; j < (channels+1)*area; j += blockDim.x) {
    const int c = j/area, point = j%area;
    const float m = mask == nullptr ? 1.0f : mask[row*area+point];
    output[row*(channels+1)*area+j] = c == channels ? m : m == 0 ? 0 :
      (float(values[row*channels*area + (nhwc ? point*channels+c : j)])-minimum[0])/denominator;
  }
}

__global__ void pack(const float* const* states, const float* contiguous, const int* actions, float* output,
                     int batch, int channels, int area, int actionPlanes, bool nhwc) {
  const int width = channels+1+actionPlanes;
  for(int j = blockIdx.x*blockDim.x+threadIdx.x; j < batch*width*area; j += blockDim.x*gridDim.x) {
    const int row = j/(width*area), k = j%(width*area);
    const int c = nhwc ? k%width : k/area, point = nhwc ? k/width : k%area;
    const float* state = contiguous ? contiguous + row*(channels+1)*area : states[row];
    output[j] = c == 0 ? state[channels*area+point] : c <= channels ? state[(c-1)*area+point] :
      float(actions[row*actionPlanes+c-channels-1] == point); // Go pass is area: an all-zero plane.
  }
}
}

void muZeroNormalize(const void* values, const float* mask, float* output,
                     int batch, int channels, int area, bool fp16, bool nhwc, cudaStream_t stream) {
  if(fp16) normalize<<<batch,256,0,stream>>>((const __half*)values,mask,output,channels,area,nhwc);
  else normalize<<<batch,256,0,stream>>>((const float*)values,mask,output,channels,area,nhwc);
}
void muZeroPack(const float* const* states, const float* contiguous, const int* actions, float* output,
                int batch, int channels, int area, int actionPlanes, bool nhwc, cudaStream_t stream) {
  const int count = batch*(channels+1+actionPlanes)*area;
  pack<<<(count+255)/256,256,0,stream>>>(states,contiguous,actions,output,batch,channels,area,actionPlanes,nhwc);
}
