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

ExcitationPulseSequence::ExcitationPulseSequence(Phantom phantom)
    : ListSequence(seq)
    , phantom(phantom)
    , dims(Vector<3,unsigned int>(phantom.texr->GetWidth(),
                                  phantom.texr->GetHeight(),
                                  phantom.texr->GetDepth()))
{
    MRIEvent e;
    
    float time = 0;

    e.action = MRIEvent::RFPULSE ;
    //e.action = MRIEvent::EXCITE ;
    //e.angleRF = Math::PI*0.5;
    //e.rfSignal = Vector<3,float>(1,0,0);
    seq.push_back(make_pair(time, e));
    float b0 = 0.5;
    float bRF = 23.51e-6;
    float w0 =  GYRO_RAD * b0;
    float w1 =  GYRO_RAD * bRF;
    float totalTime = 0.001;
    
    logger.error << totalTime << logger.end;;
    int count = 10000;
    float dt = totalTime/float(count);
    

    logger.error << dt << logger.end;

    for (int i=0;i<count;i++) {
        //e.action = MRIEvent::NONE;
        e.action = MRIEvent::RFPULSE;
        int dir = (i % 2)*2-1; // 0-1
        
        time += dt;
        

        e.rfSignal = Vector<3,float>(bRF*sin(w0 * time),0,0);
        
        seq.push_back(make_pair(time, e));

    }

    // for (int i=0;i<100;i++) {
    //     // simulate a bit
    //     e.action = MRIEvent::NONE;
    //     time += 0.001;
    //     seq.push_back(make_pair(time, e));

    // }

    // // simulate a bit
    // e.action = MRIEvent::NONE;
    // time += 4.0;
    // seq.push_back(make_pair(time, e));
    // // end

    e.action = MRIEvent::DONE;
    time += 0.1;
    seq.push_back(make_pair(time, e));

    Sort();
    
}

ExcitationPulseSequence::~ExcitationPulseSequence() {}

Vector<3,unsigned int> ExcitationPulseSequence::GetTargetDimensions() {
    return dims;
}


} // NS Science
} // NS OpenEngine

