// Visualise and test RF signals
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_RF_TESTER_H_
#define _MRI_RF_TESTER_H_

#include "IRFCoil.h"

#include <Resources/ITexture2D.h>
#include <Utils/IInspector.h>

namespace OpenEngine {
    namespace Science {
        class MathGLPlot;
    }
}
namespace MRI {
namespace Science {

using OpenEngine::Resources::ITexture2DPtr;
using OpenEngine::Science::MathGLPlot;
using OpenEngine::Utils::Inspection::ValueList;

class IFFT;
class UniformSampler;
class FrequencyData;

class RFTester {
private:
    IFFT& fft;
    IRFCoil* coil;
    MathGLPlot *timePlot, *freqPlot;
    UniformSampler* sampler;
    FrequencyData* freqData;
public:
    RFTester(IFFT& fft, IRFCoil* coil, unsigned int width, unsigned int height);
    virtual ~RFTester();

    void RunTest();

    ITexture2DPtr GetTimeTexture();
    ITexture2DPtr GetFrequencyTexture();

    void SetWindowSize(unsigned int size);
    void SetWindowPosition(unsigned int pos);

    unsigned int GetWindowSize();
    unsigned int GetWindowPosition();


    ValueList Inspect();

};

}
}

#endif //_MRI_RF_TESTER_H_
