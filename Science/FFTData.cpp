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

void FFTData::SetSampleRate(float dt) {
    max_x = 1/(dt*2);
}

pair<float,float> FFTData::GetXRange() {
    return std::make_pair(0,max_x);
}

string FFTData::GetYName() {
    return "FFT";
}

vector<float> FFTData::GetYData() {
    return convertedData;
}


}
}
