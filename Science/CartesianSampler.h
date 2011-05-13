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
#include <vector>

namespace MRI {
namespace Science {

using std::vector;

class CartesianSampler: public IMRISampler {
private:
    Vector<3, unsigned int> dims;
    vector<complex<float> > samples, images;
    IFFT* fft;
public:
    CartesianSampler(Vector<3, unsigned int> dims);
    virtual ~CartesianSampler();

    void AddSample(Vector<3,unsigned int> location, Vector<2,float> value);
    void Reset();

    Vector<3,unsigned int> GetDimensions();
    void SetDimensions(Vector<3,unsigned int> dims);
    vector<complex<float> > GetReconstructedSamples();
    vector<complex<float> > GetSamples();

};

} // NS Science
} // NS MRI

#endif // _MRI_CARTESIAN_SAMPLER_H_
