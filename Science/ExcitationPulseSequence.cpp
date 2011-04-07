// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------

#include "ExcitationPulseSequence.h"

namespace MRI {
namespace Science {

ExcitationPulseSequence::ExcitationPulseSequence(TestRFCoil* rfcoil)
    : ListSequence(seq)
    , rfcoil(rfcoil)
{
}

ExcitationPulseSequence::~ExcitationPulseSequence() {}

Vector<3,unsigned int> ExcitationPulseSequence::GetTargetDimensions() {
    return Vector<3,unsigned int>(1);
}

void ExcitationPulseSequence::Reset(MRISim& sim) {
    Clear();
    MRIEvent e;

    e.action = MRIEvent::RFPULSE | MRIEvent::GRADIENT;
    float Gz = 50e-3; // slice gradient magnitude

    e.gradient = Vector<3,float>(0.0, 0.0, 1.0); // slice normal (gradient direction)
    e.gradient.Normalize();
    e.gradient *= Gz;

    const unsigned int lobes = 6; 
    const float d = 0.001; // thickness in meters
    const float offset = 0.005; // slice plane offset from magnetic center in meters.
    const float tauPrime = 1.0 / (GYRO_HERTZ * Gz * d); // half main lobe width
    //const float flipAngle = Math::PI / 6.0;
    const float flipAngle = Math::PI * 0.5;
    const float ampl = flipAngle / (tauPrime * GYRO_RAD); // amplitude giving 90 degree pulse

    const float totalTime = tauPrime * float(lobes);
    const unsigned int steps = 5000; // number of steps
    const float dt = totalTime / float(steps);

    float w0 = sim.GetB0() * GYRO_RAD;
    rfcoil->SetDuration(totalTime);
    rfcoil->SetAmplitude(ampl);
    rfcoil->SetChannel(w0 + offset * Gz * GYRO_RAD);
    rfcoil->SetBandwidth(d * Gz * GYRO_RAD);
    
    double time = 0.0;
    for (unsigned int i = 0; i < steps; ++i) {
        e.rfSignal = rfcoil->GetSignal(time);
        seq.push_back(make_pair(time, e));

        e.action = MRIEvent::RFPULSE; // to remove gradient action
        time += dt;
    }

    e.action = MRIEvent::GRADIENT | MRIEvent::RFPULSE;
    e.gradient = -e.gradient;
    e.rfSignal = Vector<3,float>(0.0);
    seq.push_back(make_pair(time, e));

    // e.action |= MRIEvent::DONE;
    e.gradient = Vector<3,float>(0.0);
    time += totalTime * 0.5;
    seq.push_back(make_pair(time, e));

    // Sort();
}


} // NS Science
} // NS OpenEngine

