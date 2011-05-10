// MRI Cartesian Sampler
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_CARTESIAN_SAMPLER_H_
#define _MRI_CARTESIAN_SAMPLER_H_

#include "MRISim.h"
#include "CartesianFFT.h"
#include <vector>

namespace MRI {
namespace Science {

using std::vector;

class CartesianSampler: public IMRISampler {
private:
    Vector<3, unsigned int> dims;
    vector<complex<float> > samples;
    CartesianFFT* fft;
public:
    CartesianSampler(Vector<3, unsigned int> dims, bool autoWindow = true);
    virtual ~CartesianSampler();
    void AddSample(Vector<3,unsigned int> location, Vector<2,float> value);
    Vector<3,unsigned int> GetDimensions();
    FloatTexture3DPtr Reconstruct();
    void Reset();
};

} // NS Science
} // NS MRI

#endif // _MRI_CARTESIAN_SAMPLER_H_
