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

using OpenEngine::Science::IDataSet2D;

using std::vector;
using std::string;

/**
 * 
 * @class AcquisitionData AcquisitionData.h ons/MathGL/Science/AcquisitionData.h
 */
class AcquisitionData: public IDataSet2D {
private:
    float samplingDT;
    vector<float> xv, yv;
    unsigned int window, first;
public:   
    AcquisitionData(float samplingDT, float window);
    virtual ~AcquisitionData();

    string GetXName();
    string GetYName();

    void AddSample(float s);
    
    vector<float> GetXData();
    vector<float> GetYData();
};

} // NS Science
} // NS OpenEngine

#endif // _MRI_ACQUISITION_H_
