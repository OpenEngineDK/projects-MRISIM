// Visualise and test RF signals
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "RFTester.h"

#include "UniformSampler.h"
#include "FrequencyData.h"
#include <Science/MathGLPlot.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Resources;
using namespace OpenEngine::Science;
using namespace OpenEngine::Utils::Inspection;

RFTester::RFTester(IFFT& fft, 
                   IRFCoil* coil, 
                   unsigned int width, 
                   unsigned int height)
    : fft(fft)
    , coil(coil) 
    , timePlot(new MathGLPlot(width, height))
    , freqPlot(new MathGLPlot(width, height))
    , sampler(NULL)
    , freqData(NULL) 
{
    
}

RFTester::~RFTester() {
    delete timePlot;
    delete freqPlot;
}
 
void RFTester::RunTest() {
    if (sampler) delete sampler;
    if (freqData) delete freqData;

    float samplingDT = 0.000001;
    float time = 0.0;
    
    sampler = new UniformSampler(samplingDT, 0);

    unsigned int size = 0;
    while (time < coil->GetDuration()) {
        Vector<3,float> signal = coil->GetSignal(time);
        sampler->AddSample(signal[0], signal[1]);//.GetLength());
        time += samplingDT;
        size++;
    }
    sampler->SetWindowSize(size);

    freqData = new FrequencyData(sampler->GetFrequencyData(fft)); 
    freqData->SetWindowSize(size);
    timePlot->SetData(sampler);
    freqPlot->SetData(freqData);

    timePlot->Redraw();
    freqPlot->Redraw();
    
}

ITexture2DPtr RFTester::GetTimeTexture() {
    return timePlot->GetTexture();
}
    
ITexture2DPtr RFTester::GetFrequencyTexture() {
    return freqPlot->GetTexture();
}

void RFTester::SetWindowSize(unsigned int size) {
    sampler->SetWindowSize(size);
    freqData->SetWindowSize(size);
    timePlot->Redraw();
    freqPlot->Redraw();
}

void RFTester::SetWindowPosition(unsigned int pos) {
    sampler->SetWindowPosition(pos);
    freqData->SetWindowPosition(pos);
    timePlot->Redraw();
    freqPlot->Redraw();
}

unsigned int RFTester::GetWindowSize() {
    return sampler->GetWindowSize();
}
    
unsigned int RFTester::GetWindowPosition() {
    return sampler->GetWindowPosition();
}


ValueList RFTester::Inspect() {
    ValueList values;
    {
        RWValueCall<RFTester, unsigned int> *v
            = new RWValueCall<RFTester, unsigned int>(*this,
                                                      &RFTester::GetWindowSize,
                                                      &RFTester::SetWindowSize);
        v->name = "WinSz";
        v->properties[MIN] = 0;
        v->properties[MAX] = sampler->GetSize();
        v->properties[STEP] = 1;
        values.push_back(v);
    }

    {
        RWValueCall<RFTester, unsigned int> *v
            = new RWValueCall<RFTester, unsigned int>(*this,
                                                      &RFTester::GetWindowPosition,
                                                      &RFTester::SetWindowPosition);
        v->name = "WinPos";
        v->properties[MIN] = 0;
        v->properties[MAX] = sampler->GetSize();
        v->properties[STEP] = 1;
        values.push_back(v);
    }

    {
        ActionValueCall<RFTester> *v =
            new ActionValueCall<RFTester>(*this, &RFTester::RunTest);
        v->name = "Update";
        values.push_back(v);
    }
    return values;
}

}
}
