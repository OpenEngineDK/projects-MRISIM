// Data set containing frequency domain samples
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_FREQUENCY_DATA_H_
#define _MRI_FREQUENCY_DATA_H_

#include "UniformSampler.h"
#include <Science/IDataSet2D.h>

#include <vector>
#include <string>
#include <complex>

namespace MRI {
namespace Science {

using OpenEngine::Science::IDataSet2D;

using std::vector;
using std::string;
using std::complex;

/**
 * 
 * @class FrequencyData FrequencyData.h MRISIM/Science/FrequencyData.h
 */
class FrequencyData: public UniformSampler {
private:
public:   
    FrequencyData(float samplingRate, vector<complex<double> >& data);
    virtual ~FrequencyData();

    string GetXName();
    string GetYName();
};

} // NS Science
} // NS OpenEngine

#endif // _MRI_FREQUENCY_DATA_H_
