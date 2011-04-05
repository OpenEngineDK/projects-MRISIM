// Data set containing the mri data acquisition.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "UniformSampler.h"

#include "IFFT.h"
#include "FrequencyData.h"

namespace MRI {
namespace Science {

UniformSampler::UniformSampler(float samplingDT, unsigned int window)
    : samplingDT(samplingDT)
    , window(window)
    , first(0)
    , min(0.0)
    , max(0.0)
{
    
}
    
UniformSampler::~UniformSampler() {

}

string UniformSampler::GetXName() {
    return "time (s)";
}

string UniformSampler::GetYName() {
    return "signal (T)";
}

void UniformSampler::AddSample(float real) {
    if (xv.empty()) {
        xv.push_back(0.0);
        min = max = real;
    }
    else {
        xv.push_back(xv[xv.size()-1] + samplingDT);
        if (window > 0 && xv.size() > window) ++first;
        min = fmin(min, real);
        max = fmax(max, real);
    }
    yvReal.push_back(real);
    yvImag.push_back(0.0);
}

void UniformSampler::AddSample(float real, float imag) {
    if (xv.empty()) {
        xv.push_back(0.0);
        min = max = real;
    }
    else {
        xv.push_back(xv[xv.size()-1] + samplingDT);
        if (window > 0 && xv.size() > window) ++first;
        min = fmin(min, real);
        max = fmax(max, real);
    }
    yvReal.push_back(real);
    yvImag.push_back(imag);
}
    
vector<float> UniformSampler::GetXData() {
    vector<float>::iterator itr1 = xv.begin();
    for (unsigned int i = 0; i < first; ++i) {
        ++itr1;
    }
    vector<float>::iterator itr2 = itr1;
    for (unsigned int i = 0; i < window && itr2 != xv.end(); ++i) {
        ++itr2;
    }
    return vector<float>(itr1, itr2);
}

vector<float> UniformSampler::GetYData() {
    vector<float>::iterator itr1 = yvReal.begin();
    for (unsigned int i = 0; i < first; ++i) {
        ++itr1;
    }
    vector<float>::iterator itr2 = itr1;
    for (unsigned int i = 0; i < window && itr2 != yvReal.end(); ++i) {
        ++itr2;
    }
    return vector<float>(itr1, itr2);

}

FrequencyData UniformSampler::GetFrequencyData(IFFT& fft) {
    vector<float>::iterator itrReal = yvReal.begin();
    vector<float>::iterator itrImag = yvImag.begin();

    unsigned int size = yvReal.size();
    if (size > yvImag.size()) size = yvImag.size();
    vector<complex<double> > d(size);
    unsigned int i = 0;
    for (; itrReal != yvReal.end() && itrImag != yvImag.end(); ) {
        d[i] = complex<double>(*itrReal, *itrImag);
        ++i;
        ++itrReal;
        ++itrImag;
    }
    d = fft.FFT1D(d);
    return FrequencyData(1.0f/samplingDT, d);
}

void UniformSampler::SetWindowSize(unsigned int size) {
    if (window > xv.size())
        window = xv.size();
    else
        window = size;
}

void UniformSampler::SetWindowPosition(unsigned int pos) {
    if (pos + window > xv.size()) 
        first = xv.size() - window;
    else
        first = pos;
}

unsigned int UniformSampler::GetWindowSize() {
    return window;
}
    
unsigned int UniformSampler::GetWindowPosition() {
    return first;
}

unsigned int UniformSampler::GetSize() {
    return xv.size();
}

float UniformSampler::GetMax() {
    return max;
}

float UniformSampler::GetMin() {
    return min;
}

} // NS Science
} // NS OpenEngine
