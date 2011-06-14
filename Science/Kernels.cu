#include "../Resources/Phantom.h"


__constant__ float dt;
__constant__ float3 grad;
__constant__ float3 rf;
__constant__ float3 offset;
__constant__ uint3 dims;
__constant__ float3 voxelSize;


__host__ __device__ int3 idx_to_co(unsigned int idx, uint3 dim)
{
  int3 co;
  int temp = idx;
  co.x = temp%dim.x;temp -= co.x;
  co.y = (temp/(dim.x))%dim.y; temp -= co.y*dim.x;
  co.z = temp/(dim.x*dim.y);
  return co;
}

__device__ float3 RotateZ(float angle, float3 vec) {
    return make_float3(vec.x*cos(angle) + vec.y*-sin(angle),
                       vec.x*sin(angle) + vec.y*cos(angle),
                       vec.z);

}

__device__ float3 RotateX(float angle, float3 vec) {
     return make_float3(vec.x,
                        vec.y*cos(angle) + vec.z*sin(angle),
                        vec.y*-sin(angle) + vec.z*cos(angle));
}

__device__ float3 RotateY(float angle, float3 vec) {
    return make_float3(vec.x*cos(angle) + vec.z*sin(angle),
                       vec.y,
                       vec.x * -sin(angle) + vec.z * cos(angle));
}

#if USE_T_MAPS
__global__ void stepKernel(float3* magnets, float* t1, float* t2, float* eq, float* delta) {
#else
__global__ void stepKernel(float3* magnets, unsigned char* data, SpinPack* packs, float* eq, float* delta) {
#endif
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dims.x * dims.y * dims.z) return;

    #if USE_T_MAPS

    #else
    const unsigned int type = data[idx];
    if (type == 0) return;
    #endif

    const int3 co = idx_to_co(idx, dims);

    float3 magnet = magnets[idx];
    /* make_float3(magnets[3*idx+0], */
    /*                             magnets[3*idx+1], */
    /*                             magnets[3*idx+2]); */
   
    magnet = RotateX(rf.x * GYRO_RAD * dt, magnet);
    magnet = RotateY(rf.y * GYRO_RAD * dt, magnet);

    
#if USE_T_MAPS
    float dtt1 = dt/t1[idx];
    float dtt2 = dt/t2[idx];
#else
    float dtt1 = dt/packs[type].t1;
    float dtt2 = dt/packs[type].t2;
#endif

    float e1 = exp(-dtt1);
    float e2 = exp(-dtt2);
    magnet = make_float3(e2 * magnet.x,
                         e2 * magnet.y,
                         e1 * magnet.z + eq[idx] * (1.0 - e1));

    // compute voxel position
    float3 pos = make_float3((co.x + offset.x) * voxelSize.x,
                             (co.y + offset.y) * voxelSize.y,
                             (co.z + offset.z) * voxelSize.z);
        
    //compute local field
    float dG = dot(grad, pos);
    float deltaB0 = delta[idx];
    magnet = RotateZ((GYRO_RAD * dG + deltaB0) * dt, magnet);
    /* magnets[3*idx+0] = magnet.x; */
    /* magnets[3*idx+1] = magnet.y; */
    /* magnets[3*idx+2] = magnet.z; */
    magnets[idx] = magnet;
}

__global__ void invertKernel(float3* magnets) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dims.x * dims.y * dims.z) return;
    magnets[idx] = RotateX(Math::PI, magnets[idx]);
}


/*
    This version is completely unrolled.  It uses a template parameter to achieve 
    optimal code for any (power of 2) number of threads.  This requires a switch 
    statement in the host code to handle all the different thread block sizes at 
    compile time.
*/
template <unsigned int blockSize>
__global__ void
reduce(float3 *g_idata, float3 *g_odata)
{
    extern __shared__ float3 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockSize];
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; __syncthreads();}
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; __syncthreads();}
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; __syncthreads();}
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; __syncthreads();}
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; __syncthreads();}
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; __syncthreads();}
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

