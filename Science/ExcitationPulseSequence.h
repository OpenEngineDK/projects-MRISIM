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

namespace OpenEngine {
    namespace Utils {
        class PropertyTreeNode;
    }
}

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils;

/**
 * Short description.
 *
 * @class ExcitationPulseSequence ExcitationPulseSequence.h s/MRISIM/Science/ExcitationPulseSequence.h
 */
class ExcitationPulseSequence: public ListSequence {
private:
    TestRFCoil* rfcoil;
    NULLSampler sampler;

    unsigned int lobes;
    double width;
    double offset;
    double flipAngle;
    unsigned int points;
    Vector<3,float> normal;
    double Gz;
public:
    ExcitationPulseSequence(TestRFCoil* rfcoil);
    ExcitationPulseSequence(TestRFCoil* rfcoil, PropertyTreeNode* node);

    virtual ~ExcitationPulseSequence(); 
    
    IMRISampler& GetSampler();
    void Reset(MRISim& sim);
};

} // NS Science
} // NS OpenEngine

#endif // _OE_EXCITATION_PULSE_SEQUENCE_H_
