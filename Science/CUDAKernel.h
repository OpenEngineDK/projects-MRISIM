// MRI Simulator: cuda kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_CUDA_KERNEL_
#define _MRI_CUDA_KERNEL_

#include "MRISim.h"
#include "../Resources/Phantom.h"

#include <Utils/IInspector.h>
#include <Math/RandomGenerator.h>
#include <Scene/RenderNode.h>

typedef struct SpinPack {
    float t1;
    float t2;        
} SpinPack;

namespace MRI {
namespace Science {

using Resources::Phantom;
using OpenEngine::Math::RandomGenerator;
using OpenEngine::Scene::RenderNode;

class CUDAKernel: public IMRIKernel {
private:
    Phantom phantom;
    Vector<3,float>* refMagnets;
    float *eq, *deltaB0;
    Vector<3,float> gradient, rfSignal;
    unsigned char* data;
    unsigned int width, height, depth, sz, szPowTwo;
    float b0;
    RandomGenerator randomgen;

    // gpu buffers
    float* magnetsBuffer;
    unsigned char* dataBuffer;
    SpinPack* spinPackBuffer;
    float* eqBuffer;
    float* deltaBuffer;
    float *d_odata;

    inline float RandomAttribute(float base, float variance);
public:
    CUDAKernel();
    virtual ~CUDAKernel();

    void Init(Phantom phantom);
    void Step(float dt);    
    Vector<3,float>* GetMagnets() const;
    Phantom GetPhantom() const;
    void InvertSpins();
    void Reset();
    void SetGradient(Vector<3,float> gradient);
    void SetRFSignal(Vector<3,float> signal);
    Vector<3,float> GetSignal();
    Vector<3,float> GetGradient() const;
    void SetB0(float b0);
    float GetB0() const;
};


} // NS Science
} // NS OpenEngine

#endif // _MRI_CUDA_KERNEL_
