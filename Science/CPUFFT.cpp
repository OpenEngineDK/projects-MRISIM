#include "CPUFFT.h"
#include <Logging/Logger.h>
#include <fftw3.h>

namespace MRI {
namespace Science {

CPUFFT::CPUFFT() {
}

vector<complex<double > > CPUFFT::FFT1D_Real(vector<double > input) {
    vector<complex<double > > output;
    if (!input.size()) return output;


    int N1 = input.size();
    int N2 = N1/2+1;

    double *in;
    fftw_complex *out;    
    fftw_plan plan;

    in = (double*)fftw_malloc(sizeof(double) * N1);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N2);
    
    int i = 0;
    for (vector<double >::iterator itr = input.begin();
         itr != input.end();
         itr++) {
        double c = *itr;
        in[i] = c;
        ++i;
    }

    plan = fftw_plan_dft_r2c_1d(N1, in, out, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    for (int n=0;n<N2;n++) {
        output.push_back(complex<double>(out[n][0],out[n][1]));
    }

    fftw_free(in);
    fftw_free(out);


    return output;
}


vector<complex<double > > CPUFFT::FFT1D(vector<complex<double> > input) {
    if (!input.size()) return input;
    vector<complex<double > > output;

    int N = input.size();

    fftw_complex *in, *out;    
    fftw_plan plan;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    
    int i = 0;
    for (vector<complex<double > >::iterator itr = input.begin();
         itr != input.end();
         itr++) {
        complex<double> c = *itr;
        in[i][0] = c.real();
        in[i][1] = c.imag();
        ++i;
    }

    plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    for (int n=0;n<N;n++) {
        output.push_back(complex<double>(out[n][0],out[n][1]));
    }

    fftw_free(in);
    fftw_free(out);

    return output;
}

vector<complex<double> > CPUFFT::FFT2D_Inverse(vector<complex<double> > input, unsigned int w, unsigned int h, bool flip) {
    vector<complex<double> > output(w*h);
    if (!input.size()) return output;

    int rows = h;
    int cols = w;
    // int ccols = cols; ///2+1;

    fftw_complex *in, *out;
    // double *out;    
    fftw_plan plan;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * cols * rows);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * cols * rows);
    
    
    unsigned int i = 0;
    for (vector<complex<double> >::iterator itr = input.begin();
         itr != input.end();
         itr++) {
	
        //Lets get the coordinates for this pixel
        // src_co = idx_to_co(idx, dim);

        // uint2 co;
        unsigned int dI = i;
        if (flip) {
            unsigned int temp = i;
            unsigned int sX = temp%w;
            temp -= sX;
            unsigned int sY = temp/w;
            
            
            //Where should this data go?
            // T dst_co = (src_co+(dim>>1))%dim;
            unsigned int dX = (sX+(w>>1))%w;
            unsigned int dY = (sY+(h>>1))%h;
            dI = dX + dY*w;
        }
 
        complex<double> c = *itr;
        in[dI][0] = real(c);
        in[dI][1] = imag(c);
        ++i;
    }

    plan = fftw_plan_dft_2d(rows, cols,  in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for (int n = 0; n < cols * rows; ++n) {
        unsigned int dI = n;
        if (flip) {
            unsigned int temp = n;
            unsigned int sX = temp%w;
            temp -= sX;
            unsigned int sY = temp/w;
            unsigned int dX = (sX+(w>>1))%w;
            unsigned int dY = (sY+(h>>1))%h;
            dI = dX + dY*w; 
        }
        output[dI] = (complex<double>(out[n][0],out[n][1]));
    }

    fftw_free(in);
    fftw_free(out);

    return output;
}

vector<complex<double > > CPUFFT::FFT2D_Real(vector<double > input, unsigned int w, unsigned int h) {
    vector<complex<double > > output;
    if (!input.size()) return output;

    logger.info << "w " << w
                << ", h " << h
                << ", size " << input.size() 
                << ", w*h " << w*h
                << logger.end;

    //return output;

    int rows = h;
    int cols = w;
    int ccols = cols/2+1;

    double *in;
    fftw_complex *out;    
    fftw_plan plan;

    in = (double*)fftw_malloc(sizeof(double) * cols * rows);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ccols * rows);
    
    int i = 0;
    for (vector<double >::iterator itr = input.begin();
         itr != input.end();
         itr++) {
        double c = *itr;
        in[i] = c;
        ++i;
    }

    plan = fftw_plan_dft_r2c_2d(rows, cols, in, out, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    for (int n=0;n<ccols*rows;n++) {
        output.push_back(complex<double>(out[n][0],out[n][1]));
    }

    fftw_free(in);
    fftw_free(out);


    return output;
}


vector<complex<double > > CPUFFT::FFT2D(vector<complex<double > >input, unsigned int w, unsigned int h) {
    vector<complex<double > > output(w*h);
    if (!input.size()) return output;

    logger.info << "w " << w
                << ", h " << h
                << ", size " << input.size() 
                << ", w*h " << w*h
                << logger.end;

    //return output;

    int rows = h;
    int cols = w;
    // int ccols = cols/2+1;

    fftw_complex *in, *out;    
    fftw_plan plan;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * cols * rows);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * cols * rows);
    
    int i = 0;
    for (vector<complex<double> >::iterator itr = input.begin();
         itr != input.end();
         itr++) {
        complex<double> c = *itr;
        in[i][0] = c.real();
        in[i][1] = c.imag();
        ++i;
    }

    plan = fftw_plan_dft_2d(rows, cols,  in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    // plan = fftw_plan_dft_r2c_2d(rows, cols, in, out, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    for (int n=0;n<cols*rows;n++) {
        //Lets get the coordinates for this pixel
        // src_co = idx_to_co(idx, dim);

        // uint2 co;
        unsigned int temp = n;
        unsigned int sX = temp%w;
        temp -= sX;
        unsigned int sY = temp/w;
        
		//Where should this data go?
		// T dst_co = (src_co+(dim>>1))%dim;
        unsigned int dX = (sX+(w>>1))%w;
        unsigned int dY = (sY+(h>>1))%h;
        unsigned int dI = dX + dY*w;


        output[dI] = complex<double>(out[n][0], out[n][1]);
    }

    fftw_free(in);
    fftw_free(out);


    return output;
}


}
}
