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


namespace OpenEngine {
namespace Science {

using Resources::Phantom;
using MRI::Scene::SpinNode;

class CPUKernel: public IMRIKernel {
private:
    Phantom phantom;
    Vector<3,float>* magnets;
    float* eq;
    unsigned char* data;
    unsigned int width, height, depth, sz;
    float b0, gyro, time;
    Vector<3,float> signal;
public:
    CPUKernel();
    virtual ~CPUKernel();

    void Init(Phantom phantom);
    Vector<3,float> Step(float dt);    

    void RFPulse(float angle);
};


} // NS Science
} // NS OpenEngine

#endif // _MRI_CPU_KERNEL_
