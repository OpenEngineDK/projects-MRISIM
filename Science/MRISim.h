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
#include "../EventTimer.h"
#include "AcquisitionData.h"

#include <Core/IModule.h>
#include <Utils/IInspector.h>
#include <Science/MathGLPlot.h>
#include "IFFT.h"
#include "FFTData.h"

namespace MRI {
    namespace Scene {
        class SpinNode;
    }
namespace Science {

using OpenEngine::Resources::Phantom;
using Scene::SpinNode;
using namespace OpenEngine;
using namespace OpenEngine::Science;

class IMRIKernel;

class UpdateEventArg {
public:
    IMRIKernel& kernel;
    float dt, time;
    UpdateEventArg(IMRIKernel& kernel, float dt, float time)
        : kernel(kernel), dt(dt), time(time) {}
};



class MRIState {
public:
    Vector<3,float> gradient;
    float angleRF;
    enum Action {
        NONE,
        FLIP, 
        RESET
    } action; 
    MRIState()
        : gradient(Vector<3,float>(0.0)), angleRF(0.0), action(NONE) {}
    MRIState(Vector<3,float> gradient, float angleRF, Action action = NONE)
        : gradient(gradient), angleRF(angleRF), action(action) {}
    virtual ~MRIState() {};
};

class IMRISequence {
public:
    virtual ~IMRISequence() {}
    virtual MRIState GetState(float time) = 0;
};

class IMRIKernel {
protected:
public:
    virtual ~IMRIKernel() {}
    virtual void Init(Phantom phantom) = 0;
    virtual Vector<3,float> Step(float dt, float time, MRIState state) = 0;
    virtual Vector<3,float>* GetMagnets() = 0;
    virtual Phantom GetPhantom() = 0;
    virtual void RFPulse(float angle) = 0;
    virtual void Reset() = 0;
};

class MRISim : public OpenEngine::Core::IModule
             , public IListener<TimerEventArg> {
private:
    Phantom phantom;
    IMRIKernel* kernel;
    IMRISequence* sequence;
    float kernelStep, stepsPerSecond, theAccTime, theSimTime;
    bool running;
    SpinNode* spinNode;
    MathGLPlot* plot;
    MathGLPlot* fftPlot;
    EventTimer *plotTimer;
    AcquisitionData* acq;
    IFFT* fft;
    FFTData* fftData;
public:
    MRISim(Phantom phantom, IMRIKernel* kernel, IMRISequence* sequence = NULL);
    virtual ~MRISim();
    
    void Start();
    void Stop();
    void Reset();
    
    void Handle(Core::InitializeEventArg arg);
    void Handle(Core::DeinitializeEventArg arg);
    void Handle(Core::ProcessEventArg arg);
    
    void Handle(TimerEventArg arg);
    
    void SetNode(SpinNode *sn);
    void SetPlot(MathGLPlot* plot);
    void SetFFTPlot(MathGLPlot* plot);
    
    void DoFFT();
    
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
