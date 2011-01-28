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

namespace MRI {
namespace Science {

/**
 * An analytic test RF coil producing a sinc signal.
 *
 * @class TestRFCoil TestRFCoil.h MRISIM/Science/TestRFCoil.h
 */
class TestRFCoil : public IRFCoil {
private:
    float halfalpha, b1, tau, halftime;
    Vector<3,float> unitDirection;
public:
    TestRFCoil(float alpha, float b1, float tau);
    virtual ~TestRFCoil();
    Vector<3,float> GetSignal(const float time);
};

}
}

#endif //_MRI_TEST_SINC_RF_COIL_
