// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_EXCITATION_PULSE_SEQUENCE_H_
#define _OE_EXCITATION_PULSE_SEQUENCE_H_

#include "ListSequence.h"
#include "NULLSampler.h"

#include "MRISim.h"
#include "TestRFCoil.h"

namespace MRI {
namespace Science {

/**
 * Short description.
 *
 * @class ExcitationPulseSequence ExcitationPulseSequence.h s/MRISIM/Science/ExcitationPulseSequence.h
 */
class ExcitationPulseSequence: public ListSequence {
private:
    vector<pair<double, MRIEvent> > seq;
    TestRFCoil* rfcoil;
    NULLSampler sampler;
public:
    ExcitationPulseSequence(TestRFCoil* rfcoil);

    virtual ~ExcitationPulseSequence(); 

    Vector<3,unsigned int> GetTargetDimensions(); 
    
    IMRISampler& GetSampler();
    void Reset(MRISim& sim);
};

} // NS Science
} // NS OpenEngine

#endif // _OE_EXCITATION_PULSE_SEQUENCE_H_
