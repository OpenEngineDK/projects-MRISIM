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
#include "CPUFFT.h"

#include <string.h>

namespace MRI {
namespace Science {

using namespace MRI::Scene;
using namespace Utils::Inspection;

MRISim::MRISim(Phantom phantom, IMRIKernel* kernel, IMRISequence* sequence)
    : phantom(phantom)
    , kernel(kernel)
    , sequence(sequence)
    , kernelStep(42e-6)
    , stepsPerSecond(1)
    , theAccTime(0.0)
    , theSimTime(0.0)
    , running(false)
{
    kernel->Init(phantom);
    Reset();
    // Vector<3,unsigned int> dims; 
    // if (sequence)
    //     dims = sequence->GetTargetDimensions();
    // samples = vector<complex<float> >(dims[0]*dims[1]*dims[2],
    //                                   complex<float>(0.0,0.0));
}

MRISim::~MRISim() {
}

void MRISim::Start() {
    if (running) return;
    if (sequence)
        prevEvent = sequence->GetNextPoint();
    running = true;
    logger.info << "Simulator Start." << logger.end;
}
    
void MRISim::Stop() {
    if (!running) return;
    running = false;
    logger.info << "Simulator Stop." << logger.end;
}
    
void MRISim::Reset() {
    Stop();
    kernel->Reset();
    Vector<3,unsigned int> dims; 
    if(sequence) {
        sequence->Reset(*this);
        dims = sequence->GetTargetDimensions();
    }
    samples = vector<complex<float> >(dims[0]*dims[1]*dims[2],
                                      complex<float>(0.0,0.0));
    theSimTime = theAccTime = 0.0;
    logger.info << "Simulator Reset." << logger.end;
}

void MRISim::Handle(Core::InitializeEventArg arg) {
    theSimTime = theAccTime = 0.0;
}

void MRISim::Handle(Core::DeinitializeEventArg arg) {

}

void MRISim::Handle(Core::ProcessEventArg arg) {
    if (!running) return;
    theAccTime += arg.approx * 1e-6;
    double stepTime = 1.0 / stepsPerSecond;
    while (theAccTime - stepTime > 0.0) {
        theAccTime -= stepTime;
        double prevTime, nextTime;
        MRIEvent event;
        pair<double,MRIEvent> nextEvent;
        if (sequence) {
            if (sequence->HasNextPoint()) {
                nextEvent = sequence->GetNextPoint();
            }
            else {
                nextEvent = prevEvent;
                nextEvent.second.action = MRIEvent::DONE;
            }
            nextTime = nextEvent.first;
            prevTime = prevEvent.first;
            event = prevEvent.second;
            prevEvent = nextEvent;
            kernelStep = nextTime - prevTime;
        }

        if (event.action & MRIEvent::RECORD) {

            Vector<3,float> signal = kernel->GetSignal();
            sequence->GetSampler().AddSample(event.point, Vector<2,float>(signal[0], signal[1]));
            // logger.info << "Time: " << theSimTime << ", Record magnetization into grid point: " << event.point << logger.end;
            complex<double> sample = complex<double>(signal[0], signal[1]);
            unsigned int index = event.point[0] + 
                event.point[1] * phantom.texr->GetWidth() + 
                event.point[2] * phantom.texr->GetWidth() * phantom.texr->GetHeight();
                samples.at(index) = sample;
                samplesEvent.Notify(SamplesChangedEventArg(index, index+1));
        } 

        if (event.action & MRIEvent::RESET) {
            // logger.info << "Time: " << theSimTime << ", Reset magnetization to equilibrium." << logger.end;
            kernel->Reset();
        }

        if (event.action & MRIEvent::EXCITE) {
            logger.info << "Time: " << theSimTime << ", Instant excitation: " << event.angleRF * (180.0 / Math::PI) <<  " deg." << logger.end;
            kernel->RFPulse(event.angleRF, event.slice);
        }

        if (event.action & MRIEvent::GRADIENT) {
            // logger.info << "Time: " << theSimTime << ", Gradient vector set to: " << event.gradient << "." << logger.end;
            kernel->SetGradient(event.gradient);
        }

        if (event.action & MRIEvent::RFPULSE) {
            // logger.info << "Time: " << theSimTime << ", RFPulse :" << event.rfSignal << logger.end;
            kernel->SetRFSignal(event.rfSignal);
        }

        theSimTime += kernelStep;


        if (kernelStep > 0.0) {
            Timer t;
            t.Start();
            kernel->Step(kernelStep);
            t.Stop();

            logger.info << "Step took " << t.GetElapsedIntervals(1) << " us" << logger.end;
            // logger.info << "doing Kernel step: " << kernelStep << logger.end;
            stepEvent.Notify(StepEventArg(*this));
        }
        else {
            logger.info << "not doing Kernel step <= 0.0: " << kernelStep << logger.end;
        }

        // stop the simulation if next event is a DONE signal
        if (nextEvent.second.action & MRIEvent::DONE) {
            logger.info << "Time: " << theSimTime << ", Simulation sequence finished." << logger.end;
            Stop();
            break;
        }
    }
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

void MRISim::SetB0(float b0) {
    kernel->SetB0(b0);
}

float MRISim::GetB0() {
    return kernel->GetB0();
}

Phantom MRISim::GetPhantom() {
    return phantom;
}

void MRISim::SetStepsPerSecond(float steps) {
    stepsPerSecond = steps;
}

float MRISim::GetStepsPerSecond() {
    return stepsPerSecond;
}

vector<complex<float> >& MRISim::GetSamples() {
    return samples;
}

Vector<3,unsigned int> MRISim::GetSampleDimensions() {
    Vector<3,unsigned int> dims;
    if (sequence) 
        dims = sequence->GetTargetDimensions();
    return dims;
}

ValueList MRISim::Inspect() {
    ValueList values;
    {
        RValueCall<MRISim, float> *v
            = new RValueCall<MRISim,float> (*this,
                                            &MRISim::GetTime);
        v->name = "Simulation Time";
        values.push_back(v);
    }


    {
        RWValueCall<MRISim, float > *v
            = new RWValueCall<MRISim, float >(*this,
                                              &MRISim::GetStepSize,
                                              &MRISim::SetStepSize);
        v->name = "Kernel Step Size";
        v->properties[STEP] = 0.00001;
        v->properties[MIN] = 0.00001;
        v->properties[MAX] = 0.001;
        values.push_back(v);
    }

    {
        RWValueCall<MRISim, float > *v
            = new RWValueCall<MRISim, float >(*this,
                                              &MRISim::GetStepsPerSecond,
                                              &MRISim::SetStepsPerSecond);
        v->name = "Steps Per Second";
        v->properties[STEP] = 1.0;
        v->properties[MIN] = 0.1;
        v->properties[MAX] = 1000.0;
        values.push_back(v);
    }

    {
        ActionValueCall<MRISim> *v =
            new ActionValueCall<MRISim>(*this, &MRISim::Start);
        v->name = "Start";
        values.push_back(v);
    }

    {
        ActionValueCall<MRISim> *v =
            new ActionValueCall<MRISim>(*this, &MRISim::Stop);
        v->name = "Stop";
        values.push_back(v);
    }

    {
        ActionValueCall<MRISim> *v =
            new ActionValueCall<MRISim>(*this, &MRISim::Reset);
        v->name = "Reset";
        values.push_back(v);
    }

    return values;
}

} // NS Science
} // NS OpenEngine
