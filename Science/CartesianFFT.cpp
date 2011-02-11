// Reconstruct sample data based on a 2D cartesian grid FFT algorithm.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------

#include "CartesianFFT.h"

#include "IFFT.h"
#include <cstring>

namespace MRI {
namespace Science {

using namespace Resources;

CartesianFFT::CartesianFFT(IFFT& fft, vector<complex<float> >& samples, Vector<3,unsigned int> dims)
    : fft(fft) 
    , samples(samples)
    , out(samples.size(), complex<float>(0.0,0.0))
    , dims(dims)
    , size(dims[0] * dims[1] * dims[2])
{
    res = Sample3DTexturePtr(new Sample3DTexture(out, dims));
}

CartesianFFT::~CartesianFFT() {
    
}

void CartesianFFT::ReconstructSlice(unsigned int i) {
    if (size == 0 || i > dims[2]) return;
    unsigned int w = dims[0];
    unsigned int h = dims[1];

    vector<complex<double> > in(w * h);
    for (unsigned int j = 0; j < w * h; ++j)
        in[j] = samples[i * w * h + j];

    vector<complex<double> > out = fft.FFT2D(in, dims[0], dims[1]);
    
    for (unsigned int j = 0; j < w * h; ++j)
        data[i * w * h + j] = abs(out[j]);

    res->Handle(SamplesChangedEventArg(i * w * h, i * w * h + w * h));
}

FloatTexture3DPtr CartesianFFT::GetResult() {
    return res;
}
    
} // NS Science
} // NS MRI
