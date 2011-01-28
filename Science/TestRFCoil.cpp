// A test RF coil producing sinc signal.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "TestRFCoil.h"

namespace MRI {
namespace Science {


TestRFCoil::TestRFCoil(float alpha, float b1, float tau)
    : halfalpha(alpha*0.5)
    , b1(b1)
    , tau(tau)
    , halftime(tau*0.5)
{

} 

TestRFCoil::~TestRFCoil() {

}

Vector<3,float> TestRFCoil::GetSignal(const float time) {
    if (time < 0.0 || time > tau) return Vector<3,float>(0.0);
    const float reltime = time - halftime;
    if (reltime == 0.0) return unitDirection*b1; // lim(sinc(x)) = 1.0, for x -> 0.0
    const float x = halfalpha*reltime;
    return unitDirection*(b1*sinf(x)/x);
}

}
}
