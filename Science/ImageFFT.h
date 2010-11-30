// Play with spatial fourier transform
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_IMAGE_F_F_T_H_
#define _MRI_IMAGE_F_F_T_H_

#include <Resources/Texture2D.h>

namespace MRI {
namespace Science {

class IFFT;

using OpenEngine::Resources::ITexture2DPtr;
using OpenEngine::Resources::FloatTexture2DPtr;

/**
 * Play with spatial fourier transform on images.
 * @class ImageFFT ImageFFT.h MRISIM/Science/ImageFFT.h
 */
class ImageFFT {
private:
    FloatTexture2DPtr src, step1, step2, fft2d, fft2dinv;
    IFFT& fft;
public:   
    ImageFFT(ITexture2DPtr tex, IFFT& fft);
    virtual ~ImageFFT();

    ITexture2DPtr GetSrcTexture();
    ITexture2DPtr GetStep1Texture();
    ITexture2DPtr GetStep2Texture();
    ITexture2DPtr GetFFT2DTexture();
    ITexture2DPtr GetFFT2DInvTexture();
};
    
} // NS Science
} // NS MRI

#endif // _MRI_IMAGE_F_F_T_H_
