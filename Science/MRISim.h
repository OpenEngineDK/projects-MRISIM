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


class MRIEvent {
public:
    Vector<3,float> gradient;  // gradient magnetization
    float angleRF;             // pulse angle (only valid when
                               // Action::FLIP is set)
    int recX, recY, recZ;      // position of the recorded sample in
                               // k-space (only valid when
                               // Action::RECORD is set)
    enum Action {
        NONE     =    0,
        FLIP     = 1<<0, 
        RESET    = 1<<1,
        RECORD   = 1<<2,
        GRADIENT = 1<<3,
        EXCITE   = 1<<4,
        REPHASE  = 1<<5,
        LINE     = 1<<6
    }; 
    unsigned int action; 
    MRIEvent()
        : gradient(Vector<3,float>(0.0)), angleRF(0.0), recX(0), recY(0), recZ(0), action(NONE) {}
    // MRIEvent(Vector<3,float> gradient, float angleRF, unsigned int action = NONE, int recX = 0, int recY = 0, int recZ = 0)
    //     : gradient(gradient), angleRF(angleRF), recX(recX), recY(recX), recZ(recX), action(action) {}
    virtual ~MRIEvent() {};
};

class IMRISequence {
public:
    virtual ~IMRISequence() {}
    virtual MRIEvent GetEvent(float time) = 0;
    // virtual void Reset() = 0;
};

class IMRIKernel {
protected:
public:
    virtual ~IMRIKernel() {}
    virtual void Init(Phantom phantom) = 0;
    virtual Vector<3,float> Step(float dt, float time) = 0;
    virtual Vector<3,float>* GetMagnets() = 0;
    virtual Phantom GetPhantom() = 0;
    virtual void RFPulse(float angle) = 0;
    virtual void SetGradient(Vector<3,float> gradient) = 0;
    virtual void Reset() = 0;
};

class MRISim : public OpenEngine::Core::IModule
             , public IListener<TimerEventArg> {
private:
    Phantom phantom;
    IMRIKernel* kernel;
    IMRISequence* sequence;
    float kernelStep, stepsPerSecond, theAccTime, theSimTime, exciteStart, theta;
    bool running;
    SpinNode* spinNode;
    MathGLPlot* plot;
    MathGLPlot* fftPlot;
    EventTimer *plotTimer;
    AcquisitionData* acq;
    IFFT* fft;
    FFTData* fftData;
    vector<complex<double> > sliceData;
    FloatTexture2DPtr sliceTex, invTex;
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

    FloatTexture2DPtr GetKPlane();
    FloatTexture2DPtr GetImagePlane();
    
    Utils::Inspection::ValueList Inspect();
};

} // NS Science
} // NS OpenEngine

#endif // _MRI_SIMULATOR_
