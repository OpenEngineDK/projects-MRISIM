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
#include <Core/Event.h>
#include <Utils/IInspector.h>
#include <Science/MathGLPlot.h>
#include "IFFT.h"
#include "FFTData.h"

#define GYRO_HERTZ 42.576*1e6 // herz/tesla
#define GYRO_RAD GYRO_HERTZ * 2.0 * Math::PI // radians/tesla

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
    Vector<3,float> gradient, rfSignal;  // gradient and rf magnetization
    float angleRF;             // pulse angle (only valid when
                               // Action::EXCITE is set)
    unsigned int slice;        // slice to be excited. (only valid when
                               // Action::EXCITE is set) 
    Vector<3,unsigned int> point; // position of the recorded sample in
                                  // k-space (only valid when
                                  // Action::RECORD is set)
    enum Action {
        NONE     =    0, // no action
        RESET    = 1<<0, // reset magnets to equilibrium
        EXCITE   = 1<<1, // simulate rf pulse by flipping magnets instantly
        RECORD   = 1<<2, // record a sample
        GRADIENT = 1<<3, // set the gradient vector
        RFPULSE  = 1<<4, // set the rf pulse magnetization vector
        DONE     = 1<<5  // signal to stop the simulator
    }; 
    unsigned int action; 
    MRIEvent()
        : angleRF(0.0), slice(0), action(NONE) {}
    virtual ~MRIEvent() {};
};

class IMRISequence {
public:
    virtual ~IMRISequence() {}
    // virtual MRIEvent GetEvent(float time) = 0;
    virtual pair<float, MRIEvent> GetNextPoint() = 0;
    virtual void Reset() = 0;
    // tells the simulator to allocate an output array of the specified dimensions
    virtual Vector<3,unsigned int> GetTargetDimensions() = 0; 

};

class IMRIKernel {
protected:
public:
    virtual ~IMRIKernel() {}
    virtual void Init(Phantom phantom) = 0;      // initialize kernel state to reflect the given phantom
    virtual void Step(float dt, float time) = 0; // take a simulation step

    virtual Phantom GetPhantom() const = 0;            // get the current phantom
    virtual Vector<3,float> GetSignal() const = 0;     // get the current total magnetization (sum of all spins)
    virtual Vector<3,float>* GetMagnets() const = 0;   // get all the spin states (for debugging only).

    virtual void RFPulse(float angle, unsigned int slice) = 0;       // instantly rotate the vectors in a slice around the y' axis. (cheap way of simulating rf pulse).
    virtual void SetGradient(Vector<3,float> gradient) = 0; // apply a gradient vector
    virtual Vector<3,float> GetGradient() const = 0; // get the current gradient vector

    virtual void SetRFSignal(Vector<3,float> signal) = 0;   // apply an rf vector. (Slower than RFPulse, but much more precise).
    virtual void Reset() = 0;                    // reset spin states to equilibrium. (Quick an dirty way to force spin relaxation).
};

class MRISim;


struct SamplesChangedEventArg {
    SamplesChangedEventArg(unsigned int begin, unsigned int end): begin(begin), end(end) {}
    unsigned int begin, end;
};

struct StepEventArg {
    StepEventArg(MRISim& sim): sim(sim) {}
    MRISim& sim;
};

class MRISim : public OpenEngine::Core::IModule
//, public IListener<TimerEventArg> 
{
private:
    Phantom phantom;
    IMRIKernel* kernel;
    IMRISequence* sequence;
    float kernelStep, stepsPerSecond, theAccTime, theSimTime;
    bool running;
    vector<complex<float> > samples;
    pair<float,MRIEvent> prevEvent;
    Event<StepEventArg> stepEvent;
    Event<SamplesChangedEventArg> samplesEvent;
public:
    MRISim(Phantom phantom, IMRIKernel* kernel, IMRISequence* sequence = NULL);
    virtual ~MRISim();
    
    void Start();
    void Stop();
    void Reset();
    
    void Handle(Core::InitializeEventArg arg);
    void Handle(Core::DeinitializeEventArg arg);
    void Handle(Core::ProcessEventArg arg);
    
    Event<StepEventArg>& StepEvent() { return stepEvent; }
    Event<SamplesChangedEventArg>& SamplesChangedEvent() { return samplesEvent; }
    
    float GetTime();
    void SetStepSize(float);
    float GetStepSize();
    void SetStepsPerSecond(float);
    float GetStepsPerSecond();

    vector<complex<float> >& GetSamples();
    Vector<3,unsigned int> GetSampleDimensions();
    
    
    Utils::Inspection::ValueList Inspect();
};

} // NS Science
} // NS OpenEngine

#endif // _MRI_SIMULATOR_
