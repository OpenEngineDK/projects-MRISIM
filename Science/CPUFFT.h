// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_C_P_U_F_F_T_H_
#define _OE_C_P_U_F_F_T_H_

#include "IFFT.h"

namespace MRI {
namespace Science {
/**
 * Short description.
 *
 * @class CPUFFT CPUFFT.h s/MRISIM/Science/CPUFFT.h
 */
class CPUFFT : public IFFT {
private:

public:
    CPUFFT();
    vector<complex<double > > FFT1D(vector<complex<double> > input) ;
};

} // NS Science
} // NS MRI

#endif // _OE_C_P_U_F_F_T_H_
