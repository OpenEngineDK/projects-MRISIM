// Reconstruct sample data based on a 2D cartesian grid FFT algorithm.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_CARTESIAN_F_F_T_H_
#define _MRI_CARTESIAN_F_F_T_H_

#include <Math/Vector.h>
#include <Resources/Texture3D.h>
#include "../Resources/Sample3DTexture.h"

#include <vector>
#include <complex>

namespace MRI {
namespace Science {

using OpenEngine::Math::Vector;
using MRI::Resources::Sample3DTexturePtr;

using std::vector;
using std::complex;

class IFFT;

/**
 * Reconstruct sample data based on a 2D cartesian grid FFT algorithm.
 * 
 * @class CartesianFFT CartesianFFT.h MRISIM/Science/CartesianFFT.h
 */
class CartesianFFT {
private:
    IFFT& fft;
    vector<complex<float> >& samples;
    vector<complex<float> > out;
    Vector<3,unsigned int> dims;
    Sample3DTexturePtr res;
    float* data;
    unsigned int size;
public:   
    CartesianFFT(IFFT& fft, vector<complex<float> >& samples, Vector<3,unsigned int> dims);
    virtual ~CartesianFFT();

    void ReconstructSlice(unsigned int i);
    FloatTexture3DPtr GetResult();
};
    
} // NS Science
} // NS MRI

#endif //_MRI_CARTESIAN_F_F_T_H_
