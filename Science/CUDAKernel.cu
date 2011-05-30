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
        eq[i] = phantom.spinPackets[data[i]].ro*b0;
    }

    // initialize gpu memory

    float vsize[3] = {phantom.sizeX * 1e-3, phantom.sizeY* 1e-3, phantom.sizeZ* 1e-3};
    int offs[3] = {phantom.offsetX, phantom.offsetY, phantom.offsetZ};
    unsigned int _dims[3] = {width, height, depth};

    /* cudaMemcpyToSymbol(b0, &b0, sizeof(float)); */
    /* CHECK_FOR_CUDA_ERROR(); */
    cudaMemcpyToSymbol(voxelSize, vsize, 3*sizeof(float));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpyToSymbol(offset, offs, 3*sizeof(int));
    CHECK_FOR_CUDA_ERROR();

    cudaMemcpyToSymbol(dims, _dims, 3*sizeof(int));
    CHECK_FOR_CUDA_ERROR();

    cudaMemcpyToSymbol(maxSize, &sz, sizeof(int));
    CHECK_FOR_CUDA_ERROR();


    Reset();
}

void CUDAKernel::Step(float dt) {
    float _rf[3];
    float _grad[3];
    rfSignal.ToArray(_rf);
    gradient.ToArray(_grad);

    cudaMemcpyToSymbol(rf, _rf, 3*sizeof(float));
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpyToSymbol(grad, _grad, 3*sizeof(float));
    CHECK_FOR_CUDA_ERROR();

    const dim3 blockSize(32, 1, 1);
    const dim3 gridSize(sz / blockSize.x, 1);

    // start timer
    /* cutResetTimer(timer); */
	/* cutStartTimer(timer); */

    // cuda kernel call
    stepKernel<<< gridSize, blockSize >>>(dt, magnetsBuffer, dataBuffer, spinPackBuffer, eqBuffer, deltaBuffer);

	// Report timing
	/* cudaThreadSynchronize(); */
	/* cutStopTimer(timer);   */
	/* double time = cutGetTimerValue( timer );  */

	//printf("time: %.4f ms.\n", time );
    CHECK_FOR_CUDA_ERROR();
}

Vector<3,float> CUDAKernel::GetSignal() const {
    // cuda map reduce
    return Vector<3,float>();
}

void CUDAKernel::SetB0(float b0) {
    this->b0 = b0;
}

float CUDAKernel::GetB0() const {
    return b0;
}

void CUDAKernel::InvertSpins() {
    // cuda kernel invert
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
}

Vector<3,float>* CUDAKernel::GetMagnets() const {
    // copy refmagnets from gpu
    return refMagnets;
}

Phantom CUDAKernel::GetPhantom() const {
  return phantom;
}

} // NS Science
} // NS MRI
