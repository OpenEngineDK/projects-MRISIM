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
        throw Exception("ImageFFT: Only Luminance textures supported.");

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
    float* data1 = new float[sz];
    float* data1_unscaled = new float[sz];
    memset(data1, 0, sz);
    memset(data1_unscaled, 0, sz);
    step1 = FloatTexture2DPtr(new Texture2D<float>(tex->GetWidth(), tex->GetHeight(), 1, data1));
    double scale = 0.1;
    for (unsigned l = 0; l < h; ++l) {
        vector<complex<double> > line(w);
        // vector<double> line(w);
        for (unsigned int i = 0; i < w; ++i) {
            line[i] = complex<double>(data[l*w+i], 0.0);
            // line[i] = data[l*w+i];
        }
        // vector<complex<double> > fline = fft.FFT1D_Real(line);
        vector<complex<double> > fline = fft.FFT1D(line);
        // if (fline.size() != w) {
        //     logger.info << "Expected " << w << " but got " << fline.size() << "." << logger.end;
        //     throw Exception("ImageFFT: Invalid number of points returned by 1D Fourier.");
        // }
        // for (unsigned int i = 0; i < w && i < fline.size(); ++i) {
        //     data1_unscaled[l*w+i] = abs(fline[fline.size()-i]);
        //     data1[l*w+i] = scale*abs(fline[fline.size()-i]);
        // }
        // for (unsigned int i = fline.size(); i < w; ++i) {
        //     data1_unscaled[l*w+i] = abs(fline[i-fline.size()]);
        //     data1[l*w+i] = scale*abs(fline[i-fline.size()]);
        // }
        for (unsigned int i = 0; i < fline.size(); ++i) {
            data1_unscaled[l*w+i] = abs(fline[i]);
            data1[l*w+i] = scale*abs(fline[i]);
        }

    }
    
    // calculate 1d fourier for on each column (phase encoding direction)
    float* data2 = new float[w*h*sizeof(float)];
    memset(data2, 0, sz);
    step2 = FloatTexture2DPtr(new Texture2D<float>(tex->GetWidth(), tex->GetHeight(), 1, data2));

    for (unsigned r = 0; r < w; ++r) {
        // vector<complex<double> > line(w);
        vector<double> line(h);
        for (unsigned int i = 0; i < h; ++i) {
            // line[i] = complex<double>(data[l*w+i], 0.0);
            line[i] = data1[i*w+r];
        }
        vector<complex<double> > fline = fft.FFT1D_Real(line);
        for (unsigned int i = 0; i < h && i < fline.size(); ++i) {
            data2[i*w+r] = scale*abs(fline[fline.size()-i]);
        }
        for (unsigned int i = fline.size(); i < w; ++i) {
            data2[i*w+r] = scale*abs(fline[i-fline.size()]);
        }
    }

    // calculate 2d fourier 
    scale = 0.01;
    unsigned int nw = w/2+1;
    float* data2d = new float[w*h*sizeof(float)];
    float* data2d_real = new float[w*h*sizeof(float)];
    float* data2d_img = new float[w*h*sizeof(float)];
    // memset(data2d, 0, w*h*sizeof(float));
    fft2d = FloatTexture2DPtr(new Texture2D<float>(w, h, 1, data2d));
    vector<double> img(w*h);
    for (unsigned int i = 0; i < w*h; ++i) {
        img[i] = data[i];
    }
    vector<complex<double> > fimg = fft.FFT2D_Real(img, w, h);    

    for (unsigned int l = 0; l < h/2; ++l) {
        for (unsigned int i = 0; i < nw; ++i) {
            data2d_real[i+nw+l*w] = real(fimg[i+(l+h/2)*nw]);
            data2d_real[i+l*w] = real(fimg[(nw-i)+(l+h/2)*nw - 1]);
            data2d_img[i+nw+l*w] = imag(fimg[i+(l+h/2)*nw]);
            data2d_img[i+l*w] = imag(fimg[(nw-i)+(l+h/2)*nw - 1]);
            data2d[i+nw+l*w] = scale*abs(fimg[i+(l+h/2)*nw]);
            data2d[i+l*w] = scale*abs(fimg[(nw-i)+(l+h/2)*nw - 1]);
        }
    }
    for (unsigned int l = h/2; l < h; ++l) {
        for (unsigned int i = 0; i < nw; ++i) {
            data2d_real[i+nw+l*w] = real(fimg[i+(l-h/2)*nw]);
            data2d_real[i+l*w] = real(fimg[(nw-i)+(l-h/2)*nw - 1]);
            data2d_img[i+nw+l*w] = imag(fimg[i+(l-h/2)*nw]);
            data2d_img[i+l*w] = imag(fimg[(nw-i)+(l-h/2)*nw - 1]);
            data2d[i+nw+l*w] = scale*abs(fimg[i+(l-h/2)*nw]);
            data2d[i+l*w] = scale*abs(fimg[(nw-i)+(l-h/2)*nw - 1]);
        }
    }

    // calculate 2d inverse fourier
    scale = 0.00001;
    float* data2dinv = new float[w*h*sizeof(float)];
    memset(data2dinv, 0, w*h*sizeof(float));
    fft2dinv = FloatTexture2DPtr(new Texture2D<float>(w, h, 1, data2dinv));
    vector<complex<double> > invimg(w*h);
    for (unsigned int i = 0; i < w*h; ++i) {
        invimg[i] = complex<double>(data2d_real[i], data2d_img[i]);
    }
    vector<double> finvimg = fft.FFT2D_Inverse(invimg, w, h);    

    for (unsigned int i = 0; i < w*h; ++i) {
        data2dinv[i] = scale*finvimg[i];
    }
    
}

ImageFFT::~ImageFFT() {

}

ITexture2DPtr ImageFFT::GetSrcTexture() {
    return src;
}

ITexture2DPtr ImageFFT::GetStep1Texture() {
    return step1;
}

ITexture2DPtr ImageFFT::GetStep2Texture() {
    return step2;
}

ITexture2DPtr ImageFFT::GetFFT2DTexture() {
    return fft2d;
}

ITexture2DPtr ImageFFT::GetFFT2DInvTexture() {
    return fft2dinv;
}

    
} // NS Science
} // NS MRI
