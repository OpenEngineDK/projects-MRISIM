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

namespace OpenEngine {
namespace Science {

using namespace MRI::Scene;
using namespace Utils::Inspection;

MRISim::MRISim(Phantom phantom, IMRIKernel* kernel)
    : phantom(phantom)
    , kernel(kernel)
    , kernelStep(0.001)
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
    kernel->Init(phantom);    
}

void MRISim::Handle(Core::DeinitializeEventArg arg) {

}

void MRISim::Handle(Core::ProcessEventArg arg) {
    if (!running) return;
    Vector<3,float> signal = kernel->Step(kernelStep);
    if (spinNode) spinNode->M = signal;
}

void MRISim::SetNode(SpinNode *sn) {
    spinNode = sn;
}    

ValueList MRISim::Inspect() {
    ValueList values;
    // {
    //     RValueCall<MRISim, float> *v
    //         = new RValueCall<MRISim,float> (*this,
    //                                         &MRISim::GetTime);
    //     v->name = "time";
    //     values.push_back(v);
    // }


    // {
    //     RWValueCall<MRISim, float > *v
    //         = new RWValueCall<MRISim, float >(*this,
    //                                           &MRISim::GetTimeScale,
    //                                           &MRISim::SetTimeScale);
    //     v->name = "time scale";
    //     v->properties[STEP] = 0.0001;
    //     v->properties[MIN] = 0.0001;
    //     v->properties[MAX] = 0.01;
    //     values.push_back(v);
    // }
    // {
    //     RValueCall<MRISim, float> *v
    //         = new RValueCall<MRISim,float> (*this,
    //                                            &MRISim::GetTime);
    //     v->name = "time";
    //     values.push_back(v);
    // }
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
