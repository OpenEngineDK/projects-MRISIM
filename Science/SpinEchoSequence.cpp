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
    , tr(tr * 1e-3)
    , te(te * 1e-3)
    , fov(phantom.sizeX * 1e-3 * phantom.texr->GetWidth())
    , phantom(phantom)
    , dims(Vector<3,unsigned int>(phantom.texr->GetWidth(),
                           phantom.texr->GetHeight(),
                           phantom.texr->GetDepth()))
    , slice(0)
{
}

    
SpinEchoSequence::~SpinEchoSequence() {
}

Vector<3,unsigned int> SpinEchoSequence::GetTargetDimensions() {
    return dims;
}

void SpinEchoSequence::SetSlice(unsigned int slice) {
    // logger.info <<  "setslice: " << slice << logger.end;
    if (slice < phantom.texr->GetDepth()) {
        this->slice = slice;
    }
}

void SpinEchoSequence::SetFOV(float fov) {
        this->fov = fov;
}

void SpinEchoSequence::SetTR(float tr) {
        this->tr = 1e-3*tr;
}

void SpinEchoSequence::SetTE(float te) {
        this->te = 1e-3*te;
}

float SpinEchoSequence::GetFOV() {
        return fov;
}

float SpinEchoSequence::GetTR() {
        return tr*1e3;
}
    
float SpinEchoSequence::GetTE() {
        return te*1e3;
}

void SpinEchoSequence::Reset(MRISim& sim) {
    ListSequence::Clear();

    logger.info <<  "slice: " << slice << logger.end;
    float time;
    MRIEvent e;
    
    const unsigned int height = phantom.texr->GetHeight(); 
    const unsigned int width = phantom.texr->GetWidth();

    const float gyMax = 0.02;
    const float tau = float(height)/(GYRO_HERTZ * gyMax * fov);
    const float gyStart = -gyMax*0.5;
    const float dGy =  (gyMax) / float(height);
    logger.info << "dGY: " << dGy << logger.end;

    const float gx = 0.02;
    const float samplingDT = 1.0 / (fov * GYRO_HERTZ * gx);              
    const float gxDuration = samplingDT * float(width);
    logger.info << "sampling dt: " << samplingDT << logger.end;
    const float gxFirst = (gx * gxDuration * 0.5) / tau;


    float start = 0.0;
    for (unsigned int j = 0; j < height; ++j) {
        start = float(j)*tr;
        // logger.info << "start: " << start << logger.end;
        // reset + 90 degree pulse + turn on phase encoding gradient
        // turn on frequency encoding to move to the end of the x-direction
        e.action = MRIEvent::EXCITE | MRIEvent::GRADIENT;// | MRIEvent::RESET;
        e.angleRF = Math::PI*0.5;
        e.gradient = Vector<3,float>(gxFirst, gyStart + float(j)*dGy, 0.0);
        e.slice = slice;
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
            e.point = Vector<3,unsigned int>(i, height-j-1, slice);
            seq.push_back(make_pair(time, e));
            time += samplingDT;
        }
        // frequency encoding gradient off
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0);
        seq.push_back(make_pair(time, e));


        time += 0.08;
        while (time < float(j)*tr + tr) {
            e.action = MRIEvent::NONE;
            seq.push_back(make_pair(time, e));
            time += 0.08;
        }

        // start = time + 10.0 * samplingDT;
    }

    e.action = MRIEvent::DONE;
    time += 0.1;
    seq.push_back(make_pair(time, e));
    
    Sort();
}

ValueList SpinEchoSequence::Inspect() {
    ValueList values;
    
    {
        RWValueCall<SpinEchoSequence, float > *v
            = new RWValueCall<SpinEchoSequence, float >(*this,
                                              &SpinEchoSequence::GetTR,
                                              &SpinEchoSequence::SetTR);
        v->name = "TR(ms)";
        v->properties[STEP] = 1.0;
        v->properties[MIN]  = 50.0;
        v->properties[MAX]  = 4000.0;
        values.push_back(v);
    }

    {
        RWValueCall<SpinEchoSequence, float > *v
            = new RWValueCall<SpinEchoSequence, float >(*this,
                                              &SpinEchoSequence::GetTE,
                                              &SpinEchoSequence::SetTE);
        v->name = "TE(ms)";
        v->properties[STEP] = 1.0;
        v->properties[MIN]  = 0.0;
        v->properties[MAX]  = 1000.0;
        values.push_back(v);
    }

    {
        RWValueCall<SpinEchoSequence, float > *v
            = new RWValueCall<SpinEchoSequence, float >(*this,
                                              &SpinEchoSequence::GetFOV,
                                              &SpinEchoSequence::SetFOV);
        v->name = "FOV(m)";
        v->properties[STEP] = 0.01;
        v->properties[MIN]  = 0.0;
        v->properties[MAX]  = 2.0;
        values.push_back(v);
    }

    return values;
}

} // NS Science
} // NS MRI
