// Data set containing the mri data acquisition.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_ACQUISITION_DATA_H_
#define _MRI_ACQUISITION_DATA_H_

#include <Science/IDataSet2D.h>

namespace MRI {
namespace Science {

class IFFT;
class FrequencyData;

using OpenEngine::Science::IDataSet2D;

using std::vector;
using std::string;

/**
 * 
 * @class UniformSampler UniformSampler.h MRISIM/Science/UniformSampler.h
 */
class UniformSampler: public IDataSet2D {
protected:
    float samplingDT;
    vector<float> xv, yvReal, yvImag;
    unsigned int window, first;
    float min, max;
public:   
    UniformSampler(float samplingDT, unsigned int window);
    virtual ~UniformSampler();

    string GetXName();
    string GetYName();

    void AddSample(float real);
    void AddSample(float real, float imag);
    
    vector<float> GetXData();
    vector<float> GetYData();

    FrequencyData GetFrequencyData(IFFT& fft);

    void SetWindowSize(unsigned int size);
    void SetWindowPosition(unsigned int pos);

    unsigned int GetWindowSize();
    unsigned int GetWindowPosition();

    unsigned int GetSize();

    float GetMax();
    float GetMin();
};

} // NS Science
} // NS OpenEngine

#endif // _MRI_ACQUISITION_H_
