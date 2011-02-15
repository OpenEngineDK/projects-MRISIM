// MRI Simulator: cpu kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "CPUKernel.h"

#include <Math/Matrix.h>
#include <Logging/Logger.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils::Inspection;
using namespace OpenEngine::Math;

CPUKernel::CPUKernel() 
    : refMagnets(NULL)
    , labMagnets(NULL)
    , eq(NULL)
    , deltaB0(NULL)
    , gradient(Vector<3,float>(0.0))
    , rfSignal(Vector<3,float>(0.0))
    , data(NULL)
    , width(0)
    , height(0)
    , depth(0)
    , sz(0)
    , b0(0.5)
    , gyro(GYRO_RAD) // radians/Tesla
{
    randomgen.SeedWithTime();
}

CPUKernel::~CPUKernel() {
}

float CPUKernel::RandomAttribute(float base, float variance) {
    return base + randomgen.UniformFloat(-1.0,1.0) * variance;
}

void CPUKernel::Init(Phantom phantom) {
    this->phantom = phantom;
    width  = phantom.texr->GetWidth();
    height = phantom.texr->GetHeight();
    depth  = phantom.texr->GetDepth();
    sz = width*height*depth;
    refMagnets = new Vector<3,float>[sz]; 
    labMagnets = new Vector<3,float>[sz]; 
    eq = new float[sz];
    deltaB0 = new float[sz];
    data = phantom.texr->GetData();

    for (unsigned int i = 0; i < sz; ++i) {
        //deltaB0[i] = RandomAttribute(0.0, 1e-3);
        deltaB0[i] = 0.0;
        eq[i] = phantom.spinPackets[data[i]].ro*b0;
        // logger.info << "data[" << i << "] = " << int(data[i]) << logger.end;
        // logger.info << "eq[" << i << "] = " << eq[i] << logger.end;
    }
    Reset();
}

inline Vector<3,float> RotateZ(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(cos(angle), -sin(angle), 0.0,
                        sin(angle), cos(angle), 0.0,
                        0.0, 0.0, 1.0);
    return m*vec;
}

void CPUKernel::Step(float dt, float time) {
    // float T_1 = 2200.0*1e-3;
    // float T_2 = 500.0*1e-3;
    signal = Vector<3,float>();
    const float omega0 = GYRO_RAD * b0;
    const float omega0Angle = omega0*dt;
    // move rf signal into reference space
    const Vector<3,float> rf = RotateZ(-omega0Angle, rfSignal);

    for (unsigned int x = 0; x < width; ++x) {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int z = 0; z < depth; ++z) {

                unsigned int i = x + y*width + z*width*height;
                if (data[i] == 0) continue;
                
                // refMagnets[i] = RotateZ(state.angleRF, refMagnets[i]);

                // float dtt1 = dt/T_1;
                // float dtt2 = dt/T_2;
                float dtt1 = dt/phantom.spinPackets[data[i]].t1;
                float dtt2 = dt/phantom.spinPackets[data[i]].t2;
                // logger.info << "dtt1: " << dtt1 << " dtt2: " << dtt2 << logger.end;
                refMagnets[i] += Vector<3,float>(-refMagnets[i][0]*dtt2, 
                                                 -refMagnets[i][1]*dtt2, 
                                                 (eq[i]-refMagnets[i][2])*dtt1);

                float dG = gradient * Vector<3,float>(float(int(x) + phantom.offsetX) * (phantom.sizeX*1e-3),
                                                      float(int(y) + phantom.offsetY) * (phantom.sizeY*1e-3),
                                                      float(int(z) + phantom.offsetZ) * (phantom.sizeZ*1e-3));

                // logger.info << "dG: " << dG << logger.end;
                // logger.info << "angle: " << gyro * (deltaB0[i] + dG) * dt << logger.end;
                refMagnets[i] = RotateZ(GYRO_RAD * (deltaB0[i] + dG) * dt, refMagnets[i]);
                
                // add rf pulse and restore magnetization strength.
                //float len = refMagnets[i].GetLength();
                // refMagnets[i] = (refMagnets[i] + rf);
                // refMagnets[i].Normalize();
                // refMagnets[i] *= len;

                labMagnets[i] = 
                    RotateZ(omega0Angle, refMagnets[i]);
                    // Vector<3,float>(refMagnets[i][0] * cos(omega0Angle) - refMagnets[i][1] * sin(omega0Angle), 
                    //                 refMagnets[i][0] * sin(omega0Angle) + refMagnets[i][1] * cos(omega0Angle), 
                    //                 refMagnets[i][2]);
                //signal += labMagnets[i];
                signal += refMagnets[i];
            }    
        }
    }
}

void CPUKernel::Flip(unsigned int slice) {
    RFPulse(Math::PI * 0.5, slice);
}

void CPUKernel::Flop(unsigned int slice) {
    RFPulse(Math::PI, slice);
}

Vector<3,float> CPUKernel::GetSignal() const {
    return signal;
}

void CPUKernel::RFPulse(float angle, unsigned int slice) {
    Matrix<3,3,float> rot(
                          1.0, 0.0, 0.0,
                          0.0, cos(angle), sin(angle),
                          0.0,-sin(angle), cos(angle)
                          );
    // hack to only excite slice 0
    unsigned int z = slice;
    for (unsigned int i = 0; i < width; ++i) {
        for (unsigned int j = 0; j < height; ++j) {
            if (data[i + j*width  + z*width*height] == 0) continue;
            refMagnets[i + j*width + z*width*height] = rot*refMagnets[i + j*width + z*width*height];
        }
    }
}

void CPUKernel::SetGradient(Vector<3,float> gradient) {
    this->gradient = gradient;
}

Vector<3,float> CPUKernel::GetGradient() const {
    return gradient;
}

void CPUKernel::SetRFSignal(Vector<3,float> signal) {
    rfSignal = signal;
}

void CPUKernel::Reset() {
    // initialize refMagnets to b0 * spin density 
    // Signal should at all times be the sum of the spins (or not?)
    signal = Vector<3,float>();
    for (unsigned int i = 0; i < sz; ++i) {
        refMagnets[i] = labMagnets[i] = Vector<3,float>(0.0, 0.0, eq[i]);
        signal += labMagnets[i];
        //signal += refMagnets[i];
    }
    // logger.info << "Signal: " << signal << logger.end;
}

Vector<3,float>* CPUKernel::GetMagnets() const {
    //return refMagnets;
    return labMagnets;
}

Phantom CPUKernel::GetPhantom() const {
  return phantom;
}

} // NS Science
} // NS OpenEngine
