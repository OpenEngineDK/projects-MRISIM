// Data set containing frequency domain samples
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "FrequencyData.h"

namespace MRI {
namespace Science {

FrequencyData::FrequencyData(float samplingRate, vector<complex<double> >& data)
    : UniformSampler(samplingRate, 0)
{
    float accSampling = 0.0;
    min = INFINITY;
    max = -INFINITY;
    vector<complex<double> >::iterator itr = data.begin();
    for (; itr != data.end(); ++itr) {
        xv.push_back(accSampling);
        float val = abs(*itr);
        yvReal.push_back(val);
        yvImag.push_back(0.0);
        min = fmin(min, val);
        max = fmax(max, val);
        accSampling += samplingRate;
    }
}
    
FrequencyData::~FrequencyData() {

}

string FrequencyData::GetXName() {
    return "frequency";
}

string FrequencyData::GetYName() {
    return "intensity";
}

} // NS Science
} // NS OpenEngine
