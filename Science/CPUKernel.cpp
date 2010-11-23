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
    , gradient(Vector<3,float>(0.0,1e-2,0.0))
    , data(NULL)
    , width(0)
    , height(0)
    , depth(0)
    , sz(0)
    , b0(0.5)
    , gyro(42.576e06) // hz/Tesla
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
    width   = phantom.texr->GetWidth();
    height  = phantom.texr->GetHeight();
    depth   = phantom.texr->GetDepth();
    sz = width*height*depth;
    refMagnets = new Vector<3,float>[sz]; 
    labMagnets = new Vector<3,float>[sz]; 
    eq = new float[sz];
    deltaB0 = new float[sz];
    data = phantom.texr->GetData();
    
    Reset();
}

inline Vector<3,float> RotateZ(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(cos(angle), sin(angle), 0.0,
                        -sin(angle), cos(angle), 0.0,
                        0.0, 0.0, 1.0);
    return m*vec;
}

Vector<3,float> CPUKernel::Step(float dt, float time, MRIState state) {
    float T_1 = 2200/1000.0;
    float T_2 = 500/1000.0;
    signal = Vector<3,float>();
    const float omega = gyro * b0;
 
    for (unsigned int x = 0; x < width; ++x) {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int z = 0; z < depth; ++z) {
                unsigned int i = x + y*height + z*width*height;
                if (data[i] == 0) continue;
                
                // refMagnets[i] = RotateZ(state.angleRF, refMagnets[i]);

                float dtt1 = dt/T_1;
                float dtt2 = dt/T_2;
                // float dtt1 = dt/phantom.spinPackets[data[i]].t1;
                // float dtt2 = dt/phantom.spinPackets[data[i]].t2;
                // logger.info << "dtt1: " << dtt1 << " dtt2: " << dtt2 << logger.end;
                refMagnets[i] += Vector<3,float>(-refMagnets[i][0]*dtt2, 
                                                 -refMagnets[i][1]*dtt2, 
                                                 (eq[i]-refMagnets[i][2])*dtt1);
                float g = state.gradient * Vector<3,float>(float(int(x)+phantom.offsetX)*(phantom.sizeX/1000.0),
                                                     float(int(y)+phantom.offsetY)*(phantom.sizeY/1000.0),
                                                     float(int(z)+phantom.offsetZ)*(phantom.sizeZ/1000.0));
                // logger.info << "g: " << g << logger.end;
                // logger.info << "angle: " << gyro * (deltaB0[i] + g) * dt << logger.end;
                refMagnets[i] = RotateZ(gyro * (deltaB0[i] + g) * dt, refMagnets[i]);
                labMagnets[i] = 
                    Vector<3,float>(refMagnets[i][0] * cos(omega * time) - refMagnets[i][1] * sin(omega*time), 
                                    refMagnets[i][0] * sin(omega * time) + refMagnets[i][1] * cos(omega*time),  
                                    refMagnets[i][2]);
                signal += labMagnets[i];
            }    
        }
    }
    // Convert from reference to laboratory system. This should be
    // done in the for-loop, but as long as our operations are
    // distributive over addition this optimization should work just fine.
    // signal = Vector<3,float>(signal[0] * cos(omega * time) - signal[1] * sin(omega*time), 
    //                          signal[0] * sin(omega * time) + signal[1] * cos(omega*time),  
    //                          signal[2]);
    // logger.info << "Magnitude: " << signal.GetLength() << logger.end;
    return signal;
}

void CPUKernel::Flip() {
    RFPulse(Math::PI * 0.5);
}

void CPUKernel::Flop() {
    RFPulse(Math::PI);
}

void CPUKernel::RFPulse(float angle) {
    Matrix<3,3,float> rot(1.0, 0.0, 0.0,
                          0.0, cos(angle), sin(angle),
                          0.0,-sin(angle), cos(angle)
                          );

    for (unsigned int i = 0; i < sz; ++i) {
        if (data[i] == 0) continue;
        refMagnets[i] = rot*refMagnets[i];
    }
}

void CPUKernel::Reset() {
    // initialize refMagnets to b0 * spin density 
    // Signal should at all times be the sum of the spins (or not?)
    signal = Vector<3,float>();
    for (unsigned int i = 0; i < sz; ++i) {
        //deltaB0[i] = RandomAttribute(0.0, 0.5e-5);
        deltaB0[i] = 0.0;
        eq[i] = phantom.spinPackets[data[i]].ro*b0;
        refMagnets[i] = labMagnets[i] = Vector<3,float>(0.0, 0.0, eq[i]);
        signal += labMagnets[i];
    }
    // signal /= sz;
    // logger.info << "Signal: " << signal << logger.end;
}

ValueList CPUKernel::Inspect() {
    ValueList values;
    {
        ActionValueCall<CPUKernel> *v
            = new ActionValueCall<CPUKernel> (*this,
                                           &CPUKernel::Flip);
        v->name = "flip 90";
        values.push_back(v);
    }
    {
        ActionValueCall<CPUKernel> *v
            = new ActionValueCall<CPUKernel> (*this,
                                              &CPUKernel::Flop);
        v->name = "flip 180";
        values.push_back(v);
    }
    return values;

}

Vector<3,float>* CPUKernel::GetMagnets() {
  return labMagnets;
}

Phantom CPUKernel::GetPhantom() {
  return phantom;
}

} // NS Science
} // NS OpenEngine

