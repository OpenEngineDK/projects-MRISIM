// MRI Simulator
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_SIMULATOR_
#define _MRI_SIMULATOR_

#include "../Resources/Phantom.h"

#include <Core/IModule.h>
#include <Utils/IInspector.h>

namespace MRI {
    namespace Scene {
        class SpinNode;
    }
namespace Science {

using OpenEngine::Resources::Phantom;
using Scene::SpinNode;
using namespace OpenEngine;

class IMRIKernel {
public:
    virtual ~IMRIKernel() {}

    virtual void Init(Phantom phantom) = 0;
    virtual Vector<3,float> Step(float dt, float time) = 0;    
};

class MRISim: public OpenEngine::Core::IModule {
private:
    Phantom phantom;
    IMRIKernel* kernel;
    float kernelStep, stepsPerSecond, theAccTime, theSimTime;
    bool running;
    SpinNode* spinNode;
public:

    MRISim(Phantom phantom, IMRIKernel* kernel);
    virtual ~MRISim();

    void Start();
    void Stop();
    void Reset();

    void Handle(Core::InitializeEventArg arg);
    void Handle(Core::DeinitializeEventArg arg);
    void Handle(Core::ProcessEventArg arg);

    void SetNode(SpinNode *sn);

    float GetTime();
    void SetStepSize(float);
    float GetStepSize();
    void SetStepsPerSecond(float);
    float GetStepsPerSecond();

    Utils::Inspection::ValueList Inspect();
};

} // NS Science
} // NS OpenEngine

#endif // _MRI_SIMULATOR_
