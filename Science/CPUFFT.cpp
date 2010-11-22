#include "CPUFFT.h"
#include <Logging/Logger.h>
#include <fftw3.h>

namespace MRI {
namespace Science {

CPUFFT::CPUFFT() {
}

vector<complex<double > > CPUFFT::FFT1D(vector<complex<double> > input) {
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

    logger.info << "fft done" << logger.end;

    return output;
}


}
}
