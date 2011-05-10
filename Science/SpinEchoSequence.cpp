// MRI simple spin echo sequence
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "SpinEchoSequence.h"

#include "TestRFCoil.h"
#include "ExcitationPulseSequence.h"

namespace MRI {
namespace Science {

SpinEchoSequence::SpinEchoSequence(float tr, float te)
    : ListSequence(seq)
    , tr(tr * 1e-3)
    , te(te * 1e-3)
    , fov(1e-3 * 50. * 0.99) //phantom.sizeX * 1e-3 * phantom.texr->GetWidth())
    , dims(Vector<3,unsigned int>(1))
    , sampler(new CartesianSampler(dims))
{
}

    
SpinEchoSequence::~SpinEchoSequence() {
    delete sampler;
}

Vector<3,unsigned int> SpinEchoSequence::GetTargetDimensions() {
    return dims;
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
    
    TestRFCoil* rfcoil = new TestRFCoil(1.0);
    ExcitationPulseSequence* pulseSeq = new ExcitationPulseSequence(rfcoil);

    double time;
    MRIEvent e;

    Phantom phantom = sim.GetPhantom();
    dims = Vector<3,unsigned int>(phantom.texr->GetWidth(),
                                  phantom.texr->GetHeight(),
                                  phantom.texr->GetDepth());

    const unsigned int hest = 1;
    //delete sampler;
    sampler = new CartesianSampler(Vector<3,unsigned int>(dims[0] / hest, dims[1] / hest, 1));

    // this is actually number of sample in x and y direction
    const unsigned int width = sampler->GetDimensions()[0];//phantom.texr->GetWidth();
    const unsigned int height = sampler->GetDimensions()[1];//phantom.texr->GetHeight(); 
    logger.info << "sample width: " << width << logger.end;
    logger.info << "sample height: " << height << logger.end;

    const float gyMax = 20e-3; // mT/m
    const float tau = float(height)/(GYRO_HERTZ * gyMax * fov);
    const float gyStart = -gyMax*0.5;
    const float dGy =  (gyMax) / float(height);
    logger.info << "dGY: " << dGy << logger.end;

    const float gx = 20e-3; // mT/m
    const float samplingDT = 1.0 / (fov * GYRO_HERTZ * gx);              
    const float gxDuration = samplingDT * float(width);
    logger.info << "sampling dt: " << samplingDT << logger.end;
    const float gxFirst = (gx * gxDuration * 0.5) / tau;


    time = 0.0;
    double start = 0.0;
    for (unsigned int j = 0; j < height; ++j) {
        time = start = double(j) * double(tr);

        //start with reset state (full relaxation cheating)
        // e.action = MRIEvent::RESET;
        // seq.push_back(make_pair(time, e));

        // use the 90 degree flip pulse sequence
        logger.info << "tr start time: " << time << logger.end;
        pulseSeq->Reset(sim);
        while (pulseSeq->HasNextPoint()) {
            pair<double, MRIEvent> point = pulseSeq->GetNextPoint();
            // printf("rf point time before: %e\n", point.first);
            // logger.info << "rf point time before: " << point.first << logger.end;
            point.first += time;

            // printf("rf point time after: %e\n", point.first);
            // logger.info << "rf point time after: " << point.first << logger.end;
            
            seq.push_back(point);
        }
        time += pulseSeq->GetDuration();
        // logger.info << "rf done at time: " << time << logger.end;
        // setup phase encoding gradient
        time += 0.1e-3; // wait time after excitation
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(-gxFirst, gyStart + float(j)*dGy, 0.0);
        seq.push_back(make_pair(time, e));

        // turn off phase and freq encoding gradients
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0, 0.0, 0.0);
        time += tau;
        seq.push_back(make_pair(time, e));

        //180 degree pulse
        // e.action = MRIEvent::EXCITE;
        // e.angleRF = Math::PI;
        // time = start + te * 0.5;
        // seq.push_back(make_pair(time, e));
        
        // frequency encoding gradient on
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(gx, 0.0, 0.0);
        time = start + te - gxDuration * 0.5;
        seq.push_back(make_pair(time, e));
        
        // record width sample points
        for (unsigned int i = 0; i < width; ++i) {
            e.action = MRIEvent::RECORD;
            e.point = Vector<3,unsigned int>(i, height-j-1, 0);
            // logger.info << "push back record with time: " << time << logger.end;
            seq.push_back(make_pair(time, e));
            time += samplingDT;
        }

        // frequency encoding gradient off
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0);
        seq.push_back(make_pair(time, e));


        time += 0.1;
        while (time < float(j) * tr + tr) {
            e.action = MRIEvent::NONE;
            seq.push_back(make_pair(time, e));
            time += 0.1;
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

IMRISampler& SpinEchoSequence::GetSampler() {
    return *sampler;
}


} // NS Science
} // NS MRI
