// MRI simple echo planar imaging sequence
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_ECHO_PLANAR_SEQUENCE_H_
#define _MRI_ECHO_PLANAR_SEQUENCE_H_

#include "ListSequence.h"
#include "CartesianSampler.h"

#include "../Resources/Phantom.h"

#include <Utils/IInspector.h>

namespace OpenEngine {
    namespace Utils {
        class PropertyTreeNode;
    }
}

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils;
using namespace OpenEngine::Utils::Inspection;

class EchoPlanarSequence: public ListSequence {
private:
    float tr, te, fov;
    double gyMax, gx;
    // Phantom phantom;
    CartesianSampler* sampler;
    struct Slice {
        Vector<3,float> readout, phase;
        IMRISequence* excitation;
    };
    vector<Slice> slices;
    
public:
    EchoPlanarSequence(float tr, float te, float fov, Vector<3,unsigned int> dims);
    EchoPlanarSequence(PropertyTreeNode* node);
    virtual ~EchoPlanarSequence();

    // Vector<3,unsigned int> GetTargetDimensions(); 

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

#endif // _MRI_ECHO_PLANAR_SEQUENCE_
