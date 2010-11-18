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
    : magnets(NULL)
    , eq(NULL)
    , data(NULL)
    , width(0)
    , height(0)
    , depth(0)
    , sz(0)
    , b0(0.5)
    , gyro(42.576e06) // hz/Tesla
{}

CPUKernel::~CPUKernel() {
}

void CPUKernel::Init(Phantom phantom) {
    this->phantom = phantom;
    width   = phantom.texr->GetWidth();
    height  = phantom.texr->GetHeight();
    depth   = phantom.texr->GetDepth();
    sz = width*height*depth;
    magnets = new Vector<3,float>[sz];
    eq = new float[sz];
    data = phantom.texr->GetData();
    
    // initialize magnets to b0 * spin density 
    // Signal should at all times be the sum of the spins (or not?)
    signal = Vector<3,float>();
    for (unsigned int i = 0; i < sz; ++i) {
        eq[i] = phantom.spinPackets[data[i]].ro*b0;
        magnets[i] = Vector<3,float>(0.0, 0.0, eq[i]);
        signal += magnets[i];
    }
    signal /= sz;
    // logger.info << "Signal: " << signal << logger.end;
}

Vector<3,float> CPUKernel::Step(float dt, float time) {
    signal = Vector<3,float>();
    for (unsigned int i = 0; i < sz; ++i) {
        if (data[i] == 0) continue;
        float dtt1 = dt/phantom.spinPackets[data[i]].t1;
        float dtt2 = dt/phantom.spinPackets[data[i]].t2;
        // logger.info << "dtt1: " << dtt1 << " dtt2: " << dtt2 << logger.end;
        magnets[i] += Vector<3,float>(-magnets[i][0]*dtt2, 
                                      -magnets[i][1]*dtt2, 
                                      (eq[i]-magnets[i][2])*dtt1);
        signal += magnets[i];
    }    
   

    // Convert from reference to laboratory system. This should be
    // done in the for-loop, but as long as our operations are
    // distributive over addition this optimization should work just fine.
    float omega = gyro * b0;
    signal /= sz;
    signal = Vector<3,float>(signal[0] * cos(omega * time) - signal[1] * sin(omega*time), 
                             signal[0] * sin(omega * time) + signal[1] * cos(omega*time),  
                             signal[2]);
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
        magnets[i] = rot*magnets[i];
    }
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
    
} // NS Science
} // NS OpenEngine

