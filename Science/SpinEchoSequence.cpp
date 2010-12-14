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
    : ListSequence(seq), tr(tr), te(te), phantom(phantom)
{
    float time;
    MRIEvent e;
    
    // tr *= 1e-3;
    // te *= 1e-3;
    tr = 2000.0 * 1e-3;
    te = 200.0 * 1e-3;

    float gyro = 42.576; // hz/Tesla
    
    unsigned int lines = phantom.texr->GetHeight(); 
    unsigned int width = phantom.texr->GetWidth();
    float fov = phantom.sizeX*1e-3*width;      // field of view 
    
    float tau = 0.05;  // Gy duration
    float gyMaxArea = float(lines) / (2*gyro*fov);
    float gyMax = -gyMaxArea / tau; 
    float dGy =  (2*gyMax) / float(lines); //1.0 / (gyro*tau*fov);
    logger.info << "dGY: " << dGy << logger.end;

    // float gx = 0.0025;
    // float samplingDT = 1.0 / (fov * gyro * gx);              
    float samplingDT = 0.05 / (gyro);
    float gxDuration = samplingDT * float(width);
    float gx   = (gyMaxArea*2) / gxDuration;
    //float gx   = 1.0 / fov ;
    logger.info << "sampling dt: " << samplingDT << logger.end;
    //float gx = (gyMaxArea*2) * samplingDT;//1.0 / fov; //(samplingDT * fov * gyro);
    //float gx = (gyMaxArea*2) / gxDuration;
    logger.info << "gx: " << gx << logger.end;
    
    float gxFirst = gyMaxArea / tau;
    //float gxFirst = (gx * samplingDT * 0.5) / tau;

    float start = 0;//float(j)*tr;
    for (unsigned int j = 0; j < lines; ++j) {
        start = float(j)*tr + .1;
        // logger.info << "start: " << start << logger.end;
        // reset + 90 degree pulse + turn on phase encoding gradient
        // turn on frequency encoding to move to the end of the x-direction
        e.action = MRIEvent::EXCITE | MRIEvent::GRADIENT | MRIEvent::RESET;
        e.angleRF = Math::PI*0.5;
        e.gradient = Vector<3,float>(gxFirst, gyMax + float(j)*dGy, 0.0);
        time = start;
        seq.push_back(make_pair(time, e));

        // turn off phase and freq encoding gradients
        e.action = MRIEvent::GRADIENT;
        //e.gradient = Vector<3,float>(0.0, gx, 0.0);
        e.gradient = Vector<3,float>(0.0, 0.0, 0.0);
        time += tau;
        seq.push_back(make_pair(time, e));

        //180 degree pulse
        e.action = MRIEvent::REPHASE;
        e.angleRF = Math::PI;
        time = start + te*0.5;
        seq.push_back(make_pair(time, e));
        
        // frequency encoding gradient on
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(gx, 0.0, 0.0);
        time  = start + te - gxDuration * 0.5;
        seq.push_back(make_pair(time, e));
                
        e.action = MRIEvent::LINE;
        seq.push_back(make_pair(time+samplingDT, e));
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

        // start = time + 10.0 * samplingDT;
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
