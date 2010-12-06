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

SpinEchoSequence::SpinEchoSequence(float tr, float te)
    : ListSequence(seq), tr(tr), te(te)
{
    float time;
    MRIEvent e;
    
    tr *= 1e-3;
    te *= 1e-3;

    float gyro = 42.576e06; // hz/Tesla
    float fov = 0.01;      // field of view 10 mm
    float samplingDT = 0.0005;  

    unsigned int lines = 10; // hardcoded to 10x10 input
    unsigned int width = 10;
    
    float tau = 0.01;  // Gy duration
    float gyMaxArea = float(width) / (2.0*gyro*fov);
    float gyMax = gyMaxArea / tau;
    float dGy = gyMax / float(lines);

    float gxDuration = samplingDT*float(width);
    float gx = gyMax / (gxDuration*2);

    for (unsigned int j = 0; j < lines; ++j) {
        float start = float(j)*tr;
        // logger.info << "start: " << start << logger.end;
        // reset + 90 degree pulse + turn on phase encoding gradient
        e.action = MRIEvent::FLIP /*| MRIEvent::GRADIENT*/ | MRIEvent::RESET;
        e.angleRF = Math::PI*0.5;
        e.gradient = Vector<3,float>(0.0, dGy*(float(j)-0.5*float(lines)), 0.0);
        time = start;
        seq.push_back(make_pair(time, e));

        // turn off phase encoding gradient
        // turn on frequency encoding to move to the end of the x-direction
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0, gx, 0.0);
        // e.gradient = Vector<3,float>(0.0, 0.0, 0.0);
        time += tau;
        seq.push_back(make_pair(time, e));

        // turn off frequency encoding
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0);
        time += 0.5*gxDuration;
        seq.push_back(make_pair(time, e));

        //180 degree pulse
        e.action = MRIEvent::FLIP;
        e.angleRF = Math::PI;
        time = start + te*0.5;
        seq.push_back(make_pair(time, e));
        
        // frequency encoding gradient on
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(gx, 0.0, 0.0);
        time  = start + te - gxDuration*0.5;
        seq.push_back(make_pair(time, e));
                
        // record width sample points
        for (unsigned int i = 0; i < width; ++i) {
            e.action = MRIEvent::RECORD;
            e.recX = i;
            e.recY = j;
            seq.push_back(make_pair(time, e));
            time += samplingDT;
        }
        // frequency encoding gradient off
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0);
        seq.push_back(make_pair(time, e));
    }
    // for (unsigned int i = 0; i < seq.size(); ++i) {
    //     logger.info << "time: " << seq[i].first << " action: " << seq[i].second.action << logger.end;
    // }
}

    
SpinEchoSequence::~SpinEchoSequence() {
}

// MRIEvent SpinEchoSequence::GetState(float time) {
//     // logger.info << "time: " << time << logger.end;
//     MRIEvent state;
//     const float flipTime = 0.001;
//     const float echoTime = 0.02;
//     if (time > flipTime) 
//         state = MRIEvent(Vector<3,float>(0.0,0.0,1e-2), 0.0, MRIEvent::NONE);
//     if (prevTime < flipTime && flipTime <= time)
//         state = MRIEvent(Vector<3,float>(0.0,0.0,1e-2), Math::PI*0.5, MRIEvent::FLIP);
//     // if (prevTime < echoTime && echoTime <= time)
//     //     state = MRIEvent(Vector<3,float>(), Math::PI, MRIEvent::FLIP);


//     prevTime = time;
//     return state;
// }

} // NS Science
} // NS MRI
