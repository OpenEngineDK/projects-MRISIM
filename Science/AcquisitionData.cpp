// Data set containing the mri data acquisition.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "AcquisitionData.h"

namespace MRI {
namespace Science {

AcquisitionData::AcquisitionData(float samplingDT, float window)
    : samplingDT(samplingDT)
    , window(window)
    , first(0)
{
    
}
    
AcquisitionData::~AcquisitionData() {

}

string AcquisitionData::GetXName() {
    return "time (s)";
}

string AcquisitionData::GetYName() {
    return "signal (?)";
}

void AcquisitionData::AddSample(float s) {
    if (xv.empty()) xv.push_back(0.0);
    else {
        xv.push_back(xv[xv.size()-1] + samplingDT);
        if (xv.size() > window) ++first;
    }
    yv.push_back(s);
}
    
vector<float> AcquisitionData::GetXData() {
    vector<float>::iterator itr = xv.begin();
    for (unsigned int i = 0; i < first; ++i) {
        ++itr;
    }
    return vector<float>(itr, xv.end());
}

vector<float> AcquisitionData::GetYData() {
    vector<float>::iterator itr = yv.begin();
    for (unsigned int i = 0; i < first; ++i) {
        ++itr;
    }
    return vector<float>(itr, yv.end());

}

} // NS Science
} // NS OpenEngine
