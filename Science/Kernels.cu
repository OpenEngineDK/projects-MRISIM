#include "../Resources/Phantom.h"


inline __host__ __device__ uint3 idx_to_co(unsigned int idx, uint3 dim)
{
  uint3 co;
  unsigned int temp = idx;
  co.x = temp%dim.x;temp -= co.x;
  co.y = (temp/(dim.x))%dim.y; temp -= co.y*dim.x;
  co.z = temp/(dim.x*dim.y);
  return co;
}

inline __device__ float3 RotateZ(float angle, float3 vec) {
     float3 v;
     v.x = vec.x*cos(angle) + vec.y*-sin(angle);
     v.y = vec.x*sin(angle) + vec.y*cos(angle);
     v.z = vec.z;
     return v;
}

inline __device__ float3 RotateX(float angle, float3 vec) {
     float3 v;
     v.x = vec.x;
     v.y = vec.y*cos(angle) + vec.z*sin(angle);
     v.z = vec.y*-sin(angle) + vec.z*cos(angle);

     return v;
}

inline __device__ float3 RotateY(float angle, float3 vec) {
     float3 v;
     v.x = vec.x*cos(angle) + vec.z*sin(angle);
     v.y = vec.y;
     v.z = vec.x*-sin(angle) + vec.z*cos(angle);

     return v;
}


__constant__ float b0;
__constant__ float3 grad;
__constant__ float3 rf;
__constant__ int3 offset;
__constant__ unsigned int maxSize;
__constant__ uint3 dims;
__constant__ float3 voxelSize;


__global__ void stepKernel(float dt, float* magnets, unsigned char* data, SpinPack* packs, float* eq, float* delta) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= maxSize) return;
    const uint3 co = idx_to_co(idx, dims);
    const unsigned int type = data[idx];


    if (type != 0) {
        float3 magnet;
        magnet.x = magnets[idx+0];
        magnet.y = magnets[idx+1];
        magnet.z = magnets[idx+2];

        magnet = RotateX(rf.x * GYRO_RAD * dt, magnet);
        magnet = RotateY(rf.y * GYRO_RAD * dt, magnet);

        /* float dtt1 = dt/packs[type].t1; */
        /* float dtt2 = dt/packs[type].t2; */
        float e1 = exp(-dt/packs[type].t1);
        float e2 = exp(-dt/packs[type].t2);
        magnet.x = e2 * magnet.x;
        magnet.y = e2 * magnet.y;
        magnet.z = e1 * magnet.z + eq[idx] * (1.0 - e1);

        // compute voxel position
        float3 pos = make_float3(float(co.x + offset.x) * voxelSize.x,
                                 float(co.y + offset.y) * voxelSize.y,
                                 float(co.z + offset.z) * voxelSize.z);
        
        // compute local field
        float dG = dot(grad, pos);
        float deltaB0 = delta[idx];

        magnet = RotateZ(GYRO_RAD * (deltaB0 + dG) * dt, magnet);
        magnets[idx+0] = magnet.x;
        magnets[idx+1] = magnet.y;
        magnets[idx+2] = magnet.z;
    }
}
