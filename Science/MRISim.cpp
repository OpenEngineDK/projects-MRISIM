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
    , kernelStep(2e-4)
    , stepsPerSecond(1e02)
    , theAccTime(0.0)
    , theSimTime(0.0)
    , running(false)
    , spinNode(NULL)
    , plotTimer(new EventTimer(.1))
    , acq(new AcquisitionData(kernelStep, 2000))
    , fft(new CPUFFT())
    , fftData(new FFTData())
    , sliceData(vector<complex<double> >(phantom.texr->GetWidth() * phantom.texr->GetHeight()))
{
    plotTimer->TimerEvent().Attach(*this);
    kernel->Init(phantom);
    float* data = new float[phantom.texr->GetWidth() * phantom.texr->GetHeight()];
    sliceTex = FloatTexture2DPtr(new FloatTexture2D(phantom.texr->GetWidth(), phantom.texr->GetHeight(), 1, data));
    data = new float[phantom.texr->GetWidth() * phantom.texr->GetHeight()];
    invTex = FloatTexture2DPtr(new FloatTexture2D(phantom.texr->GetWidth(), phantom.texr->GetHeight(), 1,data));
}

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
}

void MRISim::Handle(Core::DeinitializeEventArg arg) {

}

void MRISim::Handle(Core::ProcessEventArg arg) {
    if (!running) return;
    theAccTime += arg.approx / 1000000.0;
    float stepTime = 1.0 / stepsPerSecond;
    while (theAccTime - stepTime > 0.0) {
        //@todo: think about when action is evaluated; before or after kernel step?
        MRIEvent event;
        if (sequence)
            event = sequence->GetEvent(theSimTime);
        if (event.action & MRIEvent::RESET)
            kernel->Reset();
        if (event.action & MRIEvent::FLIP)
            kernel->RFPulse(event.angleRF);
        if (event.action & MRIEvent::GRADIENT)
            kernel->SetGradient(event.gradient);
        theSimTime += kernelStep;
        Vector<3,float> signal = kernel->Step(kernelStep, theSimTime);
        if (event.action & MRIEvent::RECORD) {
            // record a sample in k-space
            complex<double> sample = complex<double>(signal[0], signal[1]);
            sliceData.at(event.recX + event.recY * phantom.texr->GetWidth()) 
                = sample;
            sliceTex->GetData()[event.recX + event.recY * phantom.texr->GetWidth()] = signal[0];//abs(sample);
            sliceTex->ChangedEvent().Notify(Texture2DChangedEventArg(sliceTex));
            
            // logger.info << "record value: " << signal[0] << " at (" << event.recX << ", " << event.recY << "). Time: " << theSimTime << logger.end;
            vector<double> inv = fft->FFT2D_Inverse(sliceData, sliceTex->GetWidth(), sliceTex->GetHeight());
            for (unsigned int i = 0; i < inv.size(); ++i) {
                invTex->GetData()[i] = inv[i];
                // logger.info << "transformed value: " << inv[i] << logger.end;
           }
            invTex->ChangedEvent().Notify(Texture2DChangedEventArg(invTex));
        } 
        // plot->AddPoint(theSimTime,signal[0]);
        acq->AddSample(signal[0]);
        if (spinNode) spinNode->M = signal;
        theAccTime -= stepTime;
    }
    plotTimer->Handle(arg);
}

void MRISim::Handle(TimerEventArg arg) {
    plot->Redraw();
    DoFFT();
}

void MRISim::SetNode(SpinNode *sn) {
    spinNode = sn;
}    

void MRISim::SetPlot(MathGLPlot* p) {
    plot = p;
    plot->SetData(acq);
}

void MRISim::SetFFTPlot(MathGLPlot* p) {
    fftPlot = p;
    fftPlot->SetData(fftData);
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

void MRISim::DoFFT() {    
    //vector<complex<double> > input;
    vector<double > input;
    vector<complex<double> > output;
    vector<float> data = acq->GetYData();
    for (vector<float>::iterator itr = data.begin();
         itr != data.end();
         itr++) {
        //input.push_back(complex<double>(*itr,0));
        input.push_back(*itr);
    }
    //output = fft->FFT1D(input);
    output = fft->FFT1D_Real(input);
    
    fftData->SetFFTOutput(output);
    fftData->SetSampleRate(kernelStep);

    fftPlot->Redraw();
}

FloatTexture2DPtr MRISim::GetKPlane() {
    return sliceTex;
}

FloatTexture2DPtr MRISim::GetImagePlane() {
    return invTex;
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
    {
        ActionValueCall<MRISim> *v =
            new ActionValueCall<MRISim>(*this, &MRISim::DoFFT);
        v->name = "fft";
        values.push_back(v);
    }

    return values;
}

} // NS Science
} // NS OpenEngine
