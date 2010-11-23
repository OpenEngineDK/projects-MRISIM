// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_I_F_F_T_H_
#define _OE_I_F_F_T_H_

#include <complex>
#include <vector>

using namespace std;

namespace MRI {
namespace Science {
/**
 * Short description.
 *
 * @class IFFT IFFT.h s/MRISIM/Science/IFFT.h
 */
class IFFT {
private:

public:
    virtual vector<complex<double > > FFT1D_Real(vector<double> input) =0;
    virtual vector<complex<double > > FFT1D(vector<complex<double> > input) =0;
};

} // NS Science
} // NS MRI

#endif // _OE_I_F_F_T_H_
