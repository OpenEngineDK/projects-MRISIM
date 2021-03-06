// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------

#include "ExcitationPulseSequence.h"

#include <Utils/PropertyTreeNode.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils;

ExcitationPulseSequence::ExcitationPulseSequence(TestRFCoil* rfcoil)
    : rfcoil(rfcoil)
    , lobes(6)
    , width(0.001)
    , offset(0.0)
    , flipAngle(Math::PI * 0.5)
    , points(256)
    , normal(Vector<3,float>(0.0, 0.0, 1.0))
    , Gz(50e-3)
{
}

ExcitationPulseSequence::ExcitationPulseSequence(TestRFCoil* rfcoil, PropertyTreeNode* node) 
    : rfcoil(rfcoil)
    , lobes(node->GetPath("lobes", 0))
    , width(node->GetPath("width", 0.0f))
    , offset(node->GetPath("offset", 0.0f))
    , flipAngle((node->GetPath("flip-angle", 0.0f) * Math::PI) / 180.0f)
    , points(node->GetPath("points", 0))
    , normal(node->GetPath("gradient-direction", Vector<3,float>()))
    , Gz(node->GetPath("gradient-magnitude", 0.0f))
{
    normal.Normalize();
}    

ExcitationPulseSequence::~ExcitationPulseSequence() {}

void ExcitationPulseSequence::Reset(MRISim& sim) {

    static bool log = false;

    if (log) {
        logger.info << "Reset ExcitationSequence with values" << logger.end;
        logger.info << "  lobes: " << lobes << logger.end;
        logger.info << "  slice width: " << width << logger.end;
        logger.info << "  slice offset: " << offset << logger.end;
        logger.info << "  flip angle: " << flipAngle << logger.end;
        logger.info << "  points: " << points << logger.end;
        logger.info << "  slice normal: " << normal << logger.end;
        logger.info << "  slice magnitude: " << Gz << logger.end;
    }

    Clear();
    MRIEvent e;

    e.action = MRIEvent::RFPULSE | MRIEvent::GRADIENT;
    e.gradient = normal;
    e.gradient *= Gz;

    const double tauPrime = 1.0 / (GYRO_HERTZ * Gz * width); // half main lobe width
    if (log) logger.info << "  tauPrime: " << tauPrime << logger.end;
    const double ampl = flipAngle / (tauPrime * GYRO_RAD);
    if (log) logger.info << "  amplitude: " << ampl << logger.end;

    const double totalTime = tauPrime * double(lobes);
    if (log) logger.info << "  time: " << totalTime << logger.end;

    const double dt = totalTime / double(points);
    if (log) logger.info << "  dt: " << dt << logger.end;

    const double w0 = sim.GetB0() * GYRO_RAD;
    if (log) logger.info << "  w0: " << w0 << logger.end;


    rfcoil->SetDuration(totalTime);
    rfcoil->SetAmplitude(ampl);
    rfcoil->SetChannel(offset * Gz * GYRO_RAD);
    rfcoil->SetBandwidth(width * Gz * GYRO_RAD);
    
    double time = 0.0;
    for (unsigned int i = 0; i < points; ++i) {
        e.rfSignal = rfcoil->GetSignal(time);
        seq.push_back(make_pair(time, e));

        e.action = MRIEvent::RFPULSE; // to remove gradient action
    
        //e.dt = dt;
        double t = time;
        time += dt;
        e.dt = time - t;
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

IMRISampler& ExcitationPulseSequence::GetSampler() {
    return sampler;
}

} // NS Science
} // NS OpenEngine

