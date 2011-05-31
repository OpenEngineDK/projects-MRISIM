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
    , b0(0.5) {
    randomgen.SeedWithTime();
}

CUDAKernel::~CUDAKernel() {
}

float CUDAKernel::RandomAttribute(float base, float variance) {
    return base + randomgen.UniformFloat(-1.0,1.0) * variance;
}

void CUDAKernel::Init(Phantom phantom) {
    this->phantom = phantom;
    width  = phantom.texr->GetWidth();
    height = phantom.texr->GetHeight();
    depth  = phantom.texr->GetDepth();
    sz = width*height*depth;
    refMagnets = new Vector<3,float>[sz]; 
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
    cudaMalloc((void**)&magnetsBuffer, sz * sizeof(Vector<3,float>));
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

    const dim3 blockSize(256, 1, 1);
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

Vector<3,float> CUDAKernel::GetSignal() const {
    //todo: cuda map reduce
    Vector<3,float> signal;
    cudaMemcpy((void*)refMagnets, (void*)magnetsBuffer, sz * sizeof(Vector<3,float>), cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < sz; ++i) {
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
    cudaMemcpy((void*)magnetsBuffer, (void*)refMagnets, sz * sizeof(Vector<3,float>), cudaMemcpyHostToDevice);
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
