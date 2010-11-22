// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_F_F_T_DATA_H_
#define _OE_F_F_T_DATA_H_

#include <Science/IDataSet1D.h>
#include <string>
#include <vector>
#include <complex>

namespace MRI {
namespace Science {

using std::vector;
using std::string;
using std::complex;


using OpenEngine::Science::IDataSet1D;

/**
 * Short description.
 *
 * @class FFTData FFTData.h s/MRISIM/Science/FFTData.h
 */
class  FFTData : public IDataSet1D {
private:
    vector<float> convertedData;
public:
    FFTData();
    string GetYName();
    vector<float> GetYData();
    void SetFFTOutput(vector<complex<double> > outp);
};

} // NS Science
} // NS MRI

#endif // _OE_F_F_T_DATA_H_
