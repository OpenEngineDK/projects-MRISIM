// MRI Simulator: cuda kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "CUDAKernel.h"

#include <Math/Matrix.h>
#include <Logging/Logger.h>

#include <Meta/CUDA.h>

#include "Kernels.cu"

#define REDUCE_BLOCK_SIZE 512    // must be power of two for algorithm to work!!!
#define CPU_THRESHOLD     32     // minimum number of vectors to reduce below this we simply use cpu reduction.
#define STEP_BLOCK_SIZE   256    // block size for the step kernel

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils::Inspection;
using namespace OpenEngine::Math;

CUDAKernel::CUDAKernel() 
    : refMagnets(NULL)
    , eq(NULL)
    , deltaB0(NULL)
    , gradient(Vector<3,float>(0.0))
    , rfSignal(Vector<3,float>(0.0))
    , data(NULL)
    , width(0)
    , height(0)
    , depth(0)
    , sz(0)
    , szPowTwo(0)
    , b0(0.5) {
    randomgen.SeedWithTime();
}

CUDAKernel::~CUDAKernel() {
}

float CUDAKernel::RandomAttribute(float base, float variance) {
    return base + randomgen.UniformFloat(-1.0,1.0) * variance;
}

inline unsigned int powTwo(unsigned int k) {
    if (k == 0)
        return 1;
    k--;
    for (int i=1; i<sizeof(unsigned int)*CHAR_BIT; i<<=1)
        k = k | k >> i;
    return k+1;
}

void CUDAKernel::Init(Phantom phantom) {
    this->phantom = phantom;
    width  = phantom.texr->GetWidth();
    height = phantom.texr->GetHeight();
    depth  = phantom.texr->GetDepth();
    sz = width*height*depth;
    szPowTwo = powTwo(sz);
    refMagnets = new Vector<3,float>[szPowTwo]; 
    eq = new float[sz];
    deltaB0 = new float[sz];
    data = phantom.texr->GetData();

    for (unsigned int i = 0; i < sz; ++i) {
        deltaB0[i] = 0.0;
        eq[i] = phantom.spinPackets[data[i]].ro * b0;
    }

    // initialize gpu memory

    float _voxelSize[3] = {phantom.sizeX * 1e-3, phantom.sizeY * 1e-3, phantom.sizeZ * 1e-3};
    int _offset[3] = {phantom.offsetX, phantom.offsetY, phantom.offsetZ};
    unsigned int _dims[3] = {width, height, depth};

    /* cudaMemcpyToSymbol(b0, &b0, sizeof(float)); */
    /* CHECK_FOR_CUDA_ERROR(); */
    cudaMemcpyToSymbol(voxelSize, &_voxelSize[0], 3 * sizeof(float));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpyToSymbol(offset, _offset, 3 * sizeof(int));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpyToSymbol(dims, _dims, 3 * sizeof(int));
    CHECK_FOR_CUDA_ERROR();

    // voxel data
    unsigned int sps = phantom.spinPackets.size();
    SpinPack* sp = new SpinPack[sps];
    for (int i = 0; i < sps; ++i) {
        sp[i].t1 = phantom.spinPackets[i].t1;
        sp[i].t2 = phantom.spinPackets[i].t2;
    }

    cudaMalloc((void**)&spinPackBuffer, sps * sizeof(SpinPack));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy((void*)spinPackBuffer, (void*)sp, sps * sizeof(SpinPack), cudaMemcpyHostToDevice);
    CHECK_FOR_CUDA_ERROR();
    delete sp;

    cudaMalloc((void**)&eqBuffer, sz * sizeof(float));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy((void*)eqBuffer, (void*)eq, sz * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_FOR_CUDA_ERROR();

    cudaMalloc((void**)&dataBuffer, sz * sizeof(char)); 
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy((void*)dataBuffer, (void*)data, sz * sizeof(char), cudaMemcpyHostToDevice);
    CHECK_FOR_CUDA_ERROR();

    // copy magnets in reset
    cudaMalloc((void**)&magnetsBuffer,  szPowTwo * sizeof(Vector<3,float>));
    CHECK_FOR_CUDA_ERROR();
    cudaMalloc((void**)&d_odata, (szPowTwo / REDUCE_BLOCK_SIZE * 2) * sizeof(Vector<3,float>));
    CHECK_FOR_CUDA_ERROR();

    // for now set deltab0 to all zero
    cudaMalloc((void**)&deltaBuffer, sz * sizeof(float)); 
    CHECK_FOR_CUDA_ERROR();
    cudaMemset(deltaBuffer, 0, sz * sizeof(float));
    CHECK_FOR_CUDA_ERROR();

    Reset();
}

void CUDAKernel::Step(float _dt) {
    cudaMemcpyToSymbol(rf, &rfSignal, sizeof(Vector<3,float>));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpyToSymbol(grad, &gradient, sizeof(Vector<3,float>));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpyToSymbol(dt, &_dt, sizeof(float));
    CHECK_FOR_CUDA_ERROR();

    const dim3 blockSize(STEP_BLOCK_SIZE, 1, 1);
    /* const dim3 gridSize((int)ceil((double(sz) / double(blockSize.x))), 1, 1); */
    const dim3 gridSize(sz / blockSize.x, 1, 1);

    // start timer
    /* cutResetTimer(timer); */
	/* cutStartTimer(timer); */

    // cuda kernel call
    stepKernel<<< gridSize, blockSize >>>(magnetsBuffer,
                                          dataBuffer,
                                          spinPackBuffer,
                                          eqBuffer,
                                          deltaBuffer);

    /* float* tmp = out; */
    /* out = magnetsBuffer; */
    /* magnetsBuffer = tmp; */

	// Report timing
	/* cudaThreadSynchronize(); */
	/* cutStopTimer(timer);   */
	/* double time = cutGetTimerValue( timer );  */

	//printf("time: %.4f ms.\n", time );
    CHECK_FOR_CUDA_ERROR();
}

Vector<3,float> CUDAKernel::GetSignal() {
    //todo: cuda map reduce

    int n = szPowTwo;

    float *d_idata = magnetsBuffer;
    
    while (n > CPU_THRESHOLD) {
        int threads;
        if (n == 1) 
            threads = 1;
        else
            threads = (n < REDUCE_BLOCK_SIZE*2) ? n / 2 : REDUCE_BLOCK_SIZE;
        int blocks = n / (threads * 2);

        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);        
        int smemSize = threads * sizeof(float3);

        switch (threads)
            {
            case 512:
                reduce<512><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case 256:
                reduce<256><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case 128:
                reduce<128><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case 64:
                reduce< 64><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case 32:
                reduce< 32><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case 16:
            reduce< 16><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case  8:
                reduce<  8><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case  4:
                reduce<  4><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case  2:
                reduce<  2><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            case  1:
                reduce<  1><<< dimGrid, dimBlock, smemSize >>>((float3*)d_idata, (float3*)d_odata); break;
            }
        d_idata = d_odata;
        n = blocks;
    }

    Vector<3,float> signal;
    cudaMemcpy((void*)refMagnets, (void*)d_idata, n * sizeof(Vector<3,float>), cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < n; ++i) {
        signal += refMagnets[i];
    }
    return signal;
}

void CUDAKernel::SetB0(float b0) {
    this->b0 = b0;
}

float CUDAKernel::GetB0() const {
    return b0;
}

void CUDAKernel::InvertSpins() {
    // cuda kernel invert
    const dim3 blockSize(512, 1, 1);
    const dim3 gridSize(sz / blockSize.x, 1, 1);

    invertKernel<<< gridSize, blockSize >>>(magnetsBuffer);
}

void CUDAKernel::SetGradient(Vector<3,float> gradient) {
    this->gradient = gradient;
}

Vector<3,float> CUDAKernel::GetGradient() const {
    return gradient;
}

void CUDAKernel::SetRFSignal(Vector<3,float> signal) {
    rfSignal = signal;
}

void CUDAKernel::Reset() {
    for (unsigned int i = 0; i < sz; ++i) {
        refMagnets[i] = Vector<3,float>(0.0, 0.0, eq[i]);        
    }
    cudaMemcpy((void*)magnetsBuffer, (void*)refMagnets, szPowTwo * sizeof(Vector<3,float>), cudaMemcpyHostToDevice);
    CHECK_FOR_CUDA_ERROR();
}

Vector<3,float>* CUDAKernel::GetMagnets() const {
    // copy refmagnets from gpu
    cudaMemcpy((void*)refMagnets, (void*)magnetsBuffer, sz * sizeof(Vector<3,float>), cudaMemcpyDeviceToHost);
    CHECK_FOR_CUDA_ERROR();
    return refMagnets;
}

Phantom CUDAKernel::GetPhantom() const {
  return phantom;
}

} // NS Science
} // NS MRI
