// MRI Simulator: cpu kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_CPU_KERNEL_
#define _MRI_CPU_KERNEL_

#include "MRISim.h"
#include "../Resources/Phantom.h"

#include <Utils/IInspector.h>
#include <Math/RandomGenerator.h>

namespace MRI {
namespace Science {

using Resources::Phantom;
using MRI::Scene::SpinNode;
using OpenEngine::Math::RandomGenerator;

class CPUKernel: public IMRIKernel {
private:
    Phantom phantom;
    Vector<3,float>* refMagnets, *labMagnets;
    float *eq, *deltaB0;
    Vector<3,float> gradient;
    unsigned char* data;
    unsigned int width, height, depth, sz;
    float b0, gyro;
    Vector<3,float> signal;
    RandomGenerator randomgen;

    inline float RandomAttribute(float base, float variance);
public:
    CPUKernel();
    virtual ~CPUKernel();

    void Init(Phantom phantom);
    Vector<3,float> Step(float dt, float time);    
    Vector<3,float>* GetMagnets();
    Phantom GetPhantom();
    void RFPulse(float angle);
    void Reset();
    void SetGradient(Vector<3,float> gradient);
    void Flip();
    void Flop();

    Utils::Inspection::ValueList Inspect();

};


} // NS Science
} // NS OpenEngine

#endif // _MRI_CPU_KERNEL_
