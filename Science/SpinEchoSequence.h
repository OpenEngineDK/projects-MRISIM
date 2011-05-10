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

#include "ListSequence.h"
#include "CartesianSampler.h"

#include "../Resources/Phantom.h"

#include <Utils/IInspector.h>

namespace MRI {
namespace Science {

using namespace Utils::Inspection;

class SpinEchoSequence: public ListSequence {
private:
    float tr, te, fov;
    // Phantom phantom;
    vector<pair<double, MRIEvent> > seq;
    Vector<3,unsigned int> dims;
    CartesianSampler* sampler;
public:
    SpinEchoSequence(float tr, float te);
    virtual ~SpinEchoSequence();

    Vector<3,unsigned int> GetTargetDimensions(); 

    void SetFOV(float fov);
    void SetTR(float tr);
    void SetTE(float te);
    float GetFOV();
    float GetTR();
    float GetTE();

    void Reset(MRISim& sim);
    IMRISampler& GetSampler();

    Utils::Inspection::ValueList Inspect();

};

} // NS Science
} // NS MRI

#endif // _MRI_SPIN_ECHO_SEQUENCE_
