// A test RF coil producing sinc signal.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "TestRFCoil.h"

#include <Logging/Logger.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils::Inspection;

TestRFCoil::TestRFCoil(float tau, float ampl, float chan, float bw)
    : tau(tau)
    , halftime(0.5 * tau)
    , ampl(ampl)
    , chan(chan)
    , bw(bw)
{
} 

// TestRFCoil::TestRFCoil(float alpha, float b1, float tau)
//     : halfalpha(alpha*0.5)
//     , b1(b1)
//     , tau(tau)
//     , halftime(tau*0.5)
//     , unitDirection(Vector<3,float>(1,0,0))
// {

// } 

TestRFCoil::~TestRFCoil() {

}

Vector<3,float> TestRFCoil::GetSignal(const float time) {
    if (time < 0.0 || time > tau) return Vector<3,float>(0.0);

    // float Gz = 1e-3;
    // float b0 = 0.5;
    // float bRF = 23.51e-6;
    // float w0 =  GYRO_RAD * b0;
    // float w1 =  GYRO_RAD * bRF;
    
    const float t = time - halftime;
    // const float d = 1.0;
    // const float k = GYRO_RAD * Gz * t;
    // const float kdHalf = k*d*0.5;
    // const float chan = w0;// + 2.0 * Gz * GYRO_RAD;

    //float asinc = (t == 0.0)? 1.0 : sin (kdHalf) / kdHalf; 
    // float asinc = (t == 0.0)? 1.0 : sin (Math::PI * t * d * Gz * GYRO_HERTZ) / (Math::PI * t); 
    
    float halfbw = 0.5 * bw;
    float asinc = (t == 0.0)? ampl : ampl * sin (t * halfbw) / (t * halfbw); 
    Vector<3,float> s(asinc * cos(chan * t), asinc * sin(chan * t), 0.0);
    return s;
}

float TestRFCoil::GetDuration() {
    return tau;
}

float TestRFCoil::GetAmplitude() {
    return ampl;
}

float TestRFCoil::GetChannel() {
    return chan;
}

float TestRFCoil::GetBandwidth() {
    return bw;
}

void TestRFCoil::SetAmplitude(float ampl) {
    this->ampl = ampl;
}

void TestRFCoil::SetChannel(float chan) {
    this->chan = chan;
}

void TestRFCoil::SetBandwidth(float bw) {
    this->bw = bw;
}

void TestRFCoil::SetDuration(float tau) {
    this->tau = tau;
    halftime = tau * 0.5;
}

ValueList TestRFCoil::Inspect() {
    ValueList values;

    {
        RWValueCall<TestRFCoil, float> *v
            = new RWValueCall<TestRFCoil, float>(*this,
                                                 &TestRFCoil::GetAmplitude,
                                                 &TestRFCoil::SetAmplitude);
        v->name = "amplitude (Tesla)";
        v->properties[MIN] = 1e-3; // 
        v->properties[MAX] = 50e-3; // 
        v->properties[STEP] = 2e-3; // 
        values.push_back(v);
    }

    {
        RWValueCall<TestRFCoil, float> *v
            = new RWValueCall<TestRFCoil, float>(*this,
                                                 &TestRFCoil::GetChannel,
                                                 &TestRFCoil::SetChannel);
        v->name = "channel (rad/sec)";
        v->properties[MIN] = Math::PI * 1e6; // start at 1Mhz
        v->properties[MAX] = GYRO_RAD + 8e6 * Math::PI; // range ~ 50Mhz
        v->properties[STEP] = Math::PI * 0.5e6; // step 0.5Mhz
        values.push_back(v);
    }

    {
        RWValueCall<TestRFCoil, float> *v
            = new RWValueCall<TestRFCoil, float>(*this,
                                                 &TestRFCoil::GetBandwidth,
                                                 &TestRFCoil::SetBandwidth);
        v->name = "bandwidth (rad/sec)";
        v->properties[MIN] = 1000 * Math::PI; // start at 1 khz
        v->properties[MAX] = 11000 * Math::PI; // range 10 khz
        v->properties[STEP] = Math::PI * 100; // step 100 hz
        values.push_back(v);
    }

    return values;
}


}
}
