#include "FFTData.h"


namespace MRI {
namespace Science {

FFTData::FFTData() {
    
}

void FFTData::SetFFTOutput(vector<complex<double> > data) {
    convertedData.clear();
    for(vector<complex<double> >::iterator itr = data.begin();
        itr != data.end();
        itr++) {
        complex<double> c = *itr;
        convertedData.push_back(abs(c));
    }
}

string FFTData::GetYName() {
    return "FFT";
}

vector<float> FFTData::GetYData() {
    return convertedData;
}


}
}
