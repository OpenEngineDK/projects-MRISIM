// MRI NULL Sampler
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_NULL_SAMPLER_H_
#define _MRI_NULL_SAMPLER_H_

#include "MRISim.h"

namespace MRI {
namespace Science {

class NULLSampler: public IMRISampler {
public:
    NULLSampler() {};
    virtual ~NULLSampler() {};
    void AddSample(Vector<3,unsigned int> location, Vector<2,float> value) {}
    Vector<3,unsigned int> GetDimensions() { return Vector<3,unsigned int>(); }
    vector<complex<float> > GetReconstructedSamples() { return vector<complex<float> >(); }
    vector<complex<float> > GetSamples() { return vector<complex<float> >(); }
    void Reset() {}
};

} // NS Science
} // NS MRI

#endif // _MRI_NULL_SAMPLER_H_
