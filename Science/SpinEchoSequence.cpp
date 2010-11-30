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
    
    unsigned int lines = 10;
    unsigned int width = 10;
    
    float gYMax = .2;
    float gY = gYMax / float(lines);
    float gYDuration = 0.1;
    float gX = 0.2;
    float gXDuration = 1.0;

    for (unsigned int j = 0; j < lines; ++j) {
        float start = float(j)*tr*1e-3;
        // reset + 90 degree pulse + turn on phase encoding gradient
        seq.push_back(make_pair(start, MRIEvent(Vector<3,float>(0.0, gY*(j-0.5*lines), 0.0), Math::PI*0.5, MRIEvent::FLIP /*| MRIEvent::RESET*/ | MRIEvent::GRADIENT)));

        // turn off phase encoding gradient
        // turn on frequency encoding to move to the end of the x-direction
        seq.push_back(make_pair(start+gYDuration, MRIEvent(Vector<3,float>(0.0, gX, 0.0), 0.0, MRIEvent::GRADIENT)));
        // turn off frequency encoding
        seq.push_back(make_pair(start+gYDuration+gXDuration*0.5, MRIEvent(Vector<3,float>(0.0, 0.0, 0.0), 0.0, MRIEvent::GRADIENT)));

        // 180 degree pulse
        seq.push_back(make_pair(start + te*1e-3*0.5, MRIEvent(Vector<3,float>(), Math::PI, MRIEvent::FLIP)));

        // frequency encoding gradient on
        seq.push_back(make_pair(start + te*1e-3 - gXDuration*0.5, MRIEvent(Vector<3,float>(gX, 0.0, 0.0), 0.0, MRIEvent::GRADIENT)));
        
        // record width sample points
        float gXInterval = gXDuration / float(width);
        for (unsigned int i = 0; i < width; ++i) {
            MRIEvent e(Vector<3,float>(), 0.0, MRIEvent::RECORD, 2, 2, 0);
            e.recX = i;
            e.recY = j;
            seq.push_back(make_pair(start + te*1e-3 - gXDuration * 0.5 + gXInterval + i*gXInterval, e));
        }

        // frequency encoding gradient off
        seq.push_back(make_pair(start + te*1e-3 + gXDuration*0.5, MRIEvent(Vector<3,float>(), 0.0, MRIEvent::GRADIENT)));
    }
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
