// MRI simple spin echo sequence
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "SpinEchoSequence.h"

namespace MRI {
namespace Science {

SpinEchoSequence::SpinEchoSequence()
    : prevTime(0.0)
{
        
}
    
SpinEchoSequence::~SpinEchoSequence() {
}

MRIState SpinEchoSequence::GetState(float time) {
    // logger.info << "time: " << time << logger.end;
    MRIState state;
    const float flipTime = 0.001;
    const float echoTime = 0.02;
    if (time > flipTime) 
        state = MRIState(Vector<3,float>(0.0,0.0,1e-2), 0.0, MRIState::NONE);
    if (prevTime < flipTime && flipTime <= time)
        state = MRIState(Vector<3,float>(0.0,0.0,1e-2), Math::PI*0.5, MRIState::FLIP);
    // if (prevTime < echoTime && echoTime <= time)
    //     state = MRIState(Vector<3,float>(), Math::PI, MRIState::FLIP);


    prevTime = time;
    return state;
}

} // NS Science
} // NS MRI
