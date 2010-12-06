// Play with spatial fourier transform
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "ImageFFT.h"
#include "IFFT.h"
#include <string.h>

#include <Logging/Logger.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Resources;

using namespace std;

ImageFFT::ImageFFT(ITexture2DPtr tex, IFFT& fft)
    : fft(fft)
{
    unsigned int w = tex->GetWidth();
    unsigned int h = tex->GetHeight();
    unsigned int sz = w*h*sizeof(float);

    if (tex->GetChannels() != 1)
        throw Exception("ImageFFT: Only luminance textures supported.");

    if (tex->GetType() != Types::FLOAT && tex->GetType() != Types::UBYTE)
        throw Exception("ImageFFT: Only float and char textures supported");
    
    // load the source
    float* data = new float[sz];
    src = FloatTexture2DPtr(new Texture2D<float>(w, h, 1, data));
    void* inData = tex->GetVoidDataPtr();

    if (tex->GetType() == Types::FLOAT)
        memcpy(data, inData, sz);
    else if (tex->GetType() == Types::UBYTE) {
        unsigned char* in = (unsigned char*)inData;
        for (unsigned int i = 0; i < sz; ++i)
            data[i] = float(in[i])/255.0;
    }
        
    // calculate 1d fourier for on each line
    float* data1 = new float[sz*3];
    step1 = FloatTexture2DPtr(new Texture2D<float>(tex->GetWidth(), tex->GetHeight(), 3, data1));
    for (unsigned l = 0; l < h; ++l) {
        vector<complex<double> > line(w);
        for (unsigned int i = 0; i < w; ++i) {
            line[i] = complex<double>(data[l*w+i], 0.0);
        }
        vector<complex<double> > fline = fft.FFT1D(line);
        for (unsigned int i = 0; i < fline.size(); ++i) {
            data1[l*w*3+i*3] = abs(fline[i]);
            data1[l*w*3+i*3+1] = real(fline[i]);
            data1[l*w*3+i*3+2] = imag(fline[i]);
        }
    }
    
    // calculate 1d fourier on each column (phase encoding direction)
    float* data2 = new float[w*h*sizeof(float)*3];
    step2 = FloatTexture2DPtr(new Texture2D<float>(tex->GetWidth(), tex->GetHeight(), 3, data2));

    for (unsigned r = 0; r < w; ++r) {
        vector<complex<double> > line(h);
        for (unsigned int i = 0; i < h; ++i) {
            line[i] = complex<double>(data1[i*w*3+r*3+1], data1[i*w*3+r*3+2]);
        }
        vector<complex<double> > fline = fft.FFT1D(line);
        // for (unsigned int i = 0; i < h && i < fline.size(); ++i) {
        //     data2[i*w+r] = scale*abs(fline[fline.size()-i]);
        // }
        // for (unsigned int i = fline.size(); i < w; ++i) {
        //     data2[i*w+r] = scale*abs(fline[i-fline.size()]);
        // }
        for (unsigned int i = 0; i < h; ++i) {
            data2[i*w*3+r*3] = abs(fline[i]);
            data2[i*w*3+r*3+1] = real(fline[i]);
            data2[i*w*3+r*3+2] = imag(fline[i]);
        }
    }

    // calculate 2d fourier 
    float* data2d = new float[w*h*sizeof(float)*3];
    fft2d = FloatTexture2DPtr(new Texture2D<float>(w, h, 3, data2d));
    vector<complex<double> > img(w*h);
    for (unsigned int i = 0; i < w*h; ++i) {
        img[i] = complex<double>(data[i], 0.0);
        
    }
    vector<complex<double> > fimg = fft.FFT2D(img, w, h);    
    
    for (unsigned int i = 0; i < fimg.size(); ++i) {
        data2d[i*3]   = abs(fimg[i]);   
        data2d[i*3+1] = real(fimg[i]);
        data2d[i*3+2] = imag(fimg[i]);
    }

    // calculate 2d inverse fourier
    float* data2dinv = new float[w*h*sizeof(float)];
    fft2dinv = FloatTexture2DPtr(new Texture2D<float>(w, h, 1, data2dinv));
    vector<complex<double> > invimg(w*h);
    for (unsigned int i = 0; i < w*h; ++i) {
        invimg[i] = complex<double>(data2d[i*3+1], data2d[i*3+2]);
    }
    vector<complex<double> > finvimg = fft.FFT2D_Inverse(invimg, w, h);    
    for (unsigned int i = 0; i < w*h; ++i) {
        data2dinv[i] = abs(finvimg[i]);
    }
}

ImageFFT::~ImageFFT() {

}

FloatTexture2DPtr ImageFFT::GetSrcTexture() {
    return src;
}

FloatTexture2DPtr ImageFFT::GetStep1Texture() {
    return step1;
}

FloatTexture2DPtr ImageFFT::GetStep2Texture() {
    return step2;
}

FloatTexture2DPtr ImageFFT::GetFFT2DTexture() {
    return fft2d;
}

FloatTexture2DPtr ImageFFT::GetFFT2DInvTexture() {
    return fft2dinv;
}

    
} // NS Science
} // NS MRI
