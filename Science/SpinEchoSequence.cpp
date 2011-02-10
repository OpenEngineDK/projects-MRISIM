// MRI simple spin echo sequence
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "SpinEchoSequence.h"

namespace MRI {
namespace Science {

SpinEchoSequence::SpinEchoSequence(float tr, float te, Phantom phantom)
    : ListSequence(seq)
    , tr(tr)
    , te(te)
    , phantom(phantom)
    , dims(Vector<3,unsigned int>(phantom.texr->GetWidth(),
                           phantom.texr->GetHeight(),
                           phantom.texr->GetDepth()))
{
    float time;
    MRIEvent e;
    
    tr *= 1e-3;
    te *= 1e-3;
    // tr = 2000.0 * 1e-3;
    // te = 200.0 * 1e-3;

    const float gyro = 42.576*1e6; // hz/Tesla
    
    const unsigned int lines = phantom.texr->GetHeight(); 
    const unsigned int width = phantom.texr->GetWidth();
    const float fov = 1.0*phantom.sizeX * 1e-3 * width;      // field of view 
    
    const float tau = 0.05;  // Gy duration
    const float gyMaxArea = float(lines) / (gyro*fov);
    const float gyMax = gyMaxArea / tau; 
    const float gyStart = -gyMaxArea*0.5;
    const float dGy =  (gyMax) / float(lines);
    logger.info << "dGY: " << dGy << logger.end;

    const float gx = 0.002;
    const float samplingDT = 1.0 / (fov * gyro * gx);              
    const float gxDuration = samplingDT * float(width);
    logger.info << "sampling dt: " << samplingDT << logger.end;
    
    const float gxFirst = (gx * gxDuration * 0.5) / tau;

    float start = 0;
    for (unsigned int j = 0; j < lines; ++j) {
        start = float(j)*tr + .1;
        // logger.info << "start: " << start << logger.end;
        // reset + 90 degree pulse + turn on phase encoding gradient
        // turn on frequency encoding to move to the end of the x-direction
        e.action = MRIEvent::EXCITE | MRIEvent::GRADIENT | MRIEvent::RESET;
        e.angleRF = Math::PI*0.5;
        e.gradient = Vector<3,float>(gxFirst, -gyStart + float(j)*dGy, 0.0);
        time = start;
        seq.push_back(make_pair(time, e));

        // turn off phase and freq encoding gradients
        e.action = MRIEvent::GRADIENT;
        //e.gradient = Vector<3,float>(0.0, gx, 0.0);
        e.gradient = Vector<3,float>(0.0, 0.0, 0.0);
        time += tau;
        seq.push_back(make_pair(time, e));

        //180 degree pulse
        e.action = MRIEvent::EXCITE;
        e.angleRF = Math::PI;
        time = start + te*0.5;
        seq.push_back(make_pair(time, e));
        
        // frequency encoding gradient on
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(gx, 0.0, 0.0);
        time  = start + te - gxDuration * 0.5;
        seq.push_back(make_pair(time, e));
                
        // e.action = MRIEvent::LINE;
        // seq.push_back(make_pair(time + samplingDT, e));
        // record width sample points
        for (unsigned int i = 0; i < width; ++i) {
            e.action = MRIEvent::RECORD;
            e.point = Vector<3,unsigned int>(i, j, 0);
            seq.push_back(make_pair(time, e));
            time += samplingDT;
        }
        // frequency encoding gradient off
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0);
        seq.push_back(make_pair(time, e));

        // start = time + 10.0 * samplingDT;
    }
    e.action = MRIEvent::DONE;
    time += 0.1;
    seq.push_back(make_pair(time, e));
    
    Sort();
}

    
SpinEchoSequence::~SpinEchoSequence() {
}

Vector<3,unsigned int> SpinEchoSequence::GetTargetDimensions() {
    return dims;
}

} // NS Science
} // NS MRI
