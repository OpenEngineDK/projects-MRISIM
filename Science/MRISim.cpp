// MRI Simulator
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "MRISim.h"

#include "../Scene/SpinNode.h"

namespace MRI {
namespace Science {

using namespace MRI::Scene;
using namespace Utils::Inspection;

MRISim::MRISim(Phantom phantom, IMRIKernel* kernel)
    : phantom(phantom)
    , kernel(kernel)
    , kernelStep(0.001)
    , stepsPerSecond(8.0)
    , theAccTime(0.0)
    , theSimTime(0.0)
    , running(false)
    , spinNode(NULL)
{}

MRISim::~MRISim() {
    
}

void MRISim::Start() {
    if (running) return;
    running = true;
}
    
void MRISim::Stop() {
    if (!running) return;
    running = false;
}
    
void MRISim::Reset() {
    
}

void MRISim::Handle(Core::InitializeEventArg arg) {
    theSimTime = theAccTime = 0.0;
    kernel->Init(phantom);    
}

void MRISim::Handle(Core::DeinitializeEventArg arg) {

}

void MRISim::Handle(Core::ProcessEventArg arg) {
    if (!running) return;
    theAccTime += arg.approx / 1000000.0;
    float invStep = 1.0 / stepsPerSecond;
    while (theAccTime - invStep > 0.0) {
        theSimTime += kernelStep;
        Vector<3,float> signal = kernel->Step(kernelStep, theSimTime);
        if (spinNode) spinNode->M = signal;
        theAccTime -= invStep;
    }
}

void MRISim::SetNode(SpinNode *sn) {
    spinNode = sn;
}    

float MRISim::GetTime() {
    return theSimTime;
}

void MRISim::SetStepSize(float dt) {
    kernelStep = dt;
}

float MRISim::GetStepSize() {
    return kernelStep;
}

void MRISim::SetStepsPerSecond(float steps) {
    stepsPerSecond = steps;
}

float MRISim::GetStepsPerSecond() {
    return stepsPerSecond;
}

ValueList MRISim::Inspect() {
    ValueList values;
    {
        RValueCall<MRISim, float> *v
            = new RValueCall<MRISim,float> (*this,
                                            &MRISim::GetTime);
        v->name = "time";
        values.push_back(v);
    }


    {
        RWValueCall<MRISim, float > *v
            = new RWValueCall<MRISim, float >(*this,
                                              &MRISim::GetStepSize,
                                              &MRISim::SetStepSize);
        v->name = "kernel step size (s)";
        v->properties[STEP] = 0.0001;
        v->properties[MIN] = 0.0001;
        v->properties[MAX] = 0.01;
        values.push_back(v);
    }

    {
        RWValueCall<MRISim, float > *v
            = new RWValueCall<MRISim, float >(*this,
                                              &MRISim::GetStepsPerSecond,
                                              &MRISim::SetStepsPerSecond);
        v->name = "steps per second";
        v->properties[STEP] = 1.0;
        v->properties[MIN] = 0.1;
        v->properties[MAX] = 1000.0;
        values.push_back(v);
    }

    // {
    //     RValueCall<MRISim, float> *v
    //         = new RValueCall<MRISim,float> (*this,
    //                                            &MRISim::GetTimeDT);
    //     v->name = "last dt";
    //     values.push_back(v);
    // }
    return values;

}

} // NS Science
} // NS OpenEngine
