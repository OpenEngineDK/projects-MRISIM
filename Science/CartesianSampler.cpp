// MRI Cartesian Sampler
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "CartesianSampler.h"

#include "CPUFFT.h"

namespace MRI {
namespace Science {

using namespace std;

CartesianSampler::CartesianSampler(Vector<3, unsigned int> dims, bool autoWindow)
    : dims(dims)
    , samples(vector<complex<float> >(dims[0] * dims[1] * dims[2]))
    , fft(new CartesianFFT(*(new CPUFFT()), samples, dims, autoWindow))
{
}

CartesianSampler::~CartesianSampler() {
    delete fft;
}

void CartesianSampler::AddSample(Vector<3,unsigned int> loc, Vector<2,float> value) {
    samples.at(loc[0] + loc[1] * dims[0] + loc[2] * dims[0] * dims[1]) = complex<float>(value[0], value[1]);
}

Vector<3,unsigned int> CartesianSampler::GetDimensions() {
    return dims;
}

FloatTexture3DPtr CartesianSampler::Reconstruct() {
    for (unsigned int i = 0; i < dims[2]; ++i)
        fft->ReconstructSlice(i);
    return fft->GetResult();
}

void CartesianSampler::Reset() {
    samples.clear();
    samples.resize(dims[0] * dims[1] * dims[2]);
}

} // NS Science
} // NS MRI
