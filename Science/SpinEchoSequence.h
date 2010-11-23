// MRI simple spin echo sequence
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_SPIN_ECHO_SEQUENCE_
#define _MRI_SPIN_ECHO_SEQUENCE_

#include "MRISim.h"

namespace MRI {
namespace Science {

class SpinEchoSequence: public IMRISequence {
private:
    float prevTime;
public:
    SpinEchoSequence();
    virtual ~SpinEchoSequence();
    MRIState GetState(float time);
};

} // NS Science
} // NS MRI

#endif // _MRI_SPIN_ECHO_SEQUENCE_
