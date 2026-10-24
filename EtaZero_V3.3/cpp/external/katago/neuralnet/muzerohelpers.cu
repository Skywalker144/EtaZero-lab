#include "muzerohelpers.h"
#include <cuda_fp16.h>
#include <cfloat>
namespace {
template<class T> __global__ void normalize(const T* values,const float* mask,float* output,int channels,int area,bool nhwc) {
    __shared__ float mins[256],maxs[256];
    int row=blockIdx.x,lane=threadIdx.x;float lo=FLT_MAX,hi=-FLT_MAX;
    for(int j=lane;j<channels*area;j+=blockDim.x) {
        int c=j/area,i=j%area;
        if(mask[row*area+i]>0) {
            float v=float(values[row*channels*area+(nhwc?i*channels+c:j)]);
            lo=fminf(lo,v);hi=fmaxf(hi,v);
        }
    }
    mins[lane]=lo;maxs[lane]=hi;__syncthreads();
    for(int n=128;n;n>>=1){if(lane<n){mins[lane]=fminf(mins[lane],mins[lane+n]);maxs[lane]=fmaxf(maxs[lane],maxs[lane+n]);}__syncthreads();}
    float span=maxs[0]-mins[0];float divisor=span<1e-5f?span+1e-5f:span;
    for(int j=lane;j<(channels+1)*area;j+=blockDim.x) {
        int c=j/area,i=j%area;float m=mask[row*area+i];
        float v=c==channels?m:m==0?0:(float(values[row*channels*area+(nhwc?i*channels+c:j)])-mins[0])/divisor;
        output[row*(channels+1)*area+j]=v;
    }
}
__global__ void pack(const float* const* latents,const int* from,const int* to,float* output,int rows,int channels,int area,int actionPlanes) {
    int total=rows*area*(channels+1+actionPlanes);
    for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<total;j+=blockDim.x*gridDim.x) {
        int width=channels+1+actionPlanes,c=j%width,i=(j/width)%area,row=j/width/area;
        const float* latent=latents[row];float mask=latent[channels*area+i];
        output[j]=c==0?mask:c<=channels?latent[(c-1)*area+i]*mask:
            (i==(c==channels+1?from[row]:to[row])?mask:0);
    }
}
}
void etaNormalizeLatent(const void* v,const float* m,float* out,int n,int c,int a,bool half,bool nhwc,cudaStream_t s){
    if(half)normalize<<<n,256,0,s>>>((const __half*)v,m,out,c,a,nhwc);
    else normalize<<<n,256,0,s>>>((const float*)v,m,out,c,a,nhwc);
}
void etaPackLatent(const float* const* l,const int* from,const int* to,float* out,int n,int c,int a,int p,cudaStream_t s){
    int count=n*a*(c+1+p);pack<<<(count+255)/256,256,0,s>>>(l,from,to,out,n,c,a,p);
}
