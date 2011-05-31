#include "../Resources/Phantom.h"


__constant__ float dt;
__constant__ float3 grad;
__constant__ float3 rf;
__constant__ int3 offset;
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

__global__ void stepKernel(float* magnets, unsigned char* data, SpinPack* packs, float* eq, float* delta) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dims.x * dims.y * dims.z) return;
    const unsigned int type = data[idx];
    if (type == 0) return;

    const int3 co = idx_to_co(idx, dims);

    float3 magnet = make_float3(magnets[3*idx+0],
                                magnets[3*idx+1],
                                magnets[3*idx+2]);
   
    magnet = RotateX(rf.x * GYRO_RAD * dt, magnet);
    magnet = RotateY(rf.y * GYRO_RAD * dt, magnet);

    float dtt1 = dt/packs[type].t1;
    float dtt2 = dt/packs[type].t2;

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
    magnet = RotateZ(GYRO_RAD * (deltaB0 + dG) * dt, magnet);
    magnets[3*idx+0] = magnet.x;
    magnets[3*idx+1] = magnet.y;
    magnets[3*idx+2] = magnet.z;
}

__global__ void invertKernel(float* magnets) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dims.x * dims.y * dims.z) return;
    magnets[3*idx+0] = -magnets[3*idx+0];
    magnets[3*idx+1] = -magnets[3*idx+1];
    magnets[3*idx+2] = -magnets[3*idx+2];
}
