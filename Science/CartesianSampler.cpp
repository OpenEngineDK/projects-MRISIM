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

CartesianSampler::CartesianSampler(Vector<3, unsigned int> dims)
    : dims(dims)
    , samples(vector<complex<float> >(dims[0] * dims[1] * dims[2]))
    , images(vector<complex<float> >(dims[0] * dims[1] * dims[2]))
    , fft(new CPUFFT())
{
}

CartesianSampler::~CartesianSampler() {
    delete fft;
}

void CartesianSampler::AddSample(Vector<3,unsigned int> loc, Vector<2,float> value) {
    unsigned int index = loc[0] + loc[1] * dims[0] + loc[2] * dims[0] * dims[1];
    samples.at(index) = complex<float>(value[0], value[1]);
    samplesEvent.Notify(SamplesChangedEventArg(samples, dims, index, index+1));
}

void CartesianSampler::SetDimensions(Vector<3,unsigned int> dims) {
    this->dims = dims;
    unsigned int size = dims[0] * dims[1] * dims[2];
    samples.resize(size);
    images.resize(size);
}

Vector<3,unsigned int> CartesianSampler::GetDimensions() {
    return dims;
}

vector<complex<float> > CartesianSampler::GetReconstructedSamples() {

    // todo: only reconstruct when a slice has changed.
    // reconstruct slice by slice.
    for (unsigned int k = 0; k < dims[2]; ++k) {
        unsigned int w = dims[0];
        unsigned int h = dims[1];
        
        // copy a slice
        vector<complex<double> > in(w * h);
        for (unsigned int i = 0; i < w * h; ++i)
            in[i] = samples[k * w * h + i];
        
        //do an inverse 2d fourier
        vector<complex<double> > image = fft->FFT2D_Inverse(in, dims[0], dims[1], true);
        
        //write back reconstructed slice.
        for (unsigned int i = 0; i < w * h; ++i) {
            images[k * w * h + i] = image[i];
        }
    }
    return images;
}

vector<complex<float> > CartesianSampler::GetSamples() {
    return samples;
}


void CartesianSampler::Reset() {
    unsigned int size = dims[0] * dims[1] * dims[2];
    samples = vector<complex<float> >(size);
    images = vector<complex<float> >(size);
    samplesEvent.Notify(SamplesChangedEventArg(samples, dims, 0, size));
}

} // NS Science
} // NS MRI
