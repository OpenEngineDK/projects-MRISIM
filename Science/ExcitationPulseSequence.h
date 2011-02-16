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

namespace MRI {
namespace Science {

/**
 * Short description.
 *
 * @class ExcitationPulseSequence ExcitationPulseSequence.h s/MRISIM/Science/ExcitationPulseSequence.h
 */
class ExcitationPulseSequence: public ListSequence {
private:
    Phantom phantom;
    vector<pair<float, MRIEvent> > seq;
    Vector<3,unsigned int> dims;

public:
    ExcitationPulseSequence(Phantom phantom);

    virtual ~ExcitationPulseSequence(); 

    Vector<3,unsigned int> GetTargetDimensions(); 
    
};

} // NS Science
} // NS OpenEngine

#endif // _OE_EXCITATION_PULSE_SEQUENCE_H_
