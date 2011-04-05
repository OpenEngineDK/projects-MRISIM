// A test RF coil producing sinc signal.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_TEST_SINC_RF_COIL_
#define _MRI_TEST_SINC_RF_COIL_

#include "IRFCoil.h"
#include "MRISim.h"

#include <Utils/IInspector.h>

namespace MRI {
namespace Science {

using OpenEngine::Utils::Inspection::ValueList;

/**
 * An analytic test RF coil producing a slice selective sinc signal.
 *
 * @class TestRFCoil TestRFCoil.h MRISIM/Science/TestRFCoil.h
 */
class TestRFCoil : public IRFCoil {
private:
    float halfalpha, b1, tau, halftime;
    float ampl, chan, bw;
    Vector<3,float> unitDirection;
public:
    TestRFCoil(float tau, float ampl = 1.0, float chan = 0.0, float bw = 1.0);
    // TestRFCoil(float alpha, float b1, float tau);
    virtual ~TestRFCoil();
    Vector<3,float> GetSignal(const float time);
    float GetDuration();
    float GetAmplitude();
    float GetChannel();
    float GetBandwidth();

    void SetAmplitude(float ampl);
    void SetChannel(float chan);
    void SetBandwidth(float bw);
    void SetDuration(float duration);
   
    ValueList Inspect();
    
};

}
}

#endif //_MRI_TEST_SINC_RF_COIL_
