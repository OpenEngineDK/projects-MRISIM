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

#include <Utils/PropertyTreeNode.h>

#include <Math/Quaternion.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils;
using namespace OpenEngine::Math;

SpinEchoSequence::SpinEchoSequence(float tr, float te, float fov, Vector<3,unsigned int> dims)
    : tr(tr * 1e-3)
    , te(te * 1e-3)
    , fov(fov) 
    , sampler(new CartesianSampler(dims))
{
    slices = vector<Slice>(1);
    TestRFCoil* rfcoil = new TestRFCoil(1.0);
    slices[0].readout = Vector<3,float>(1.0, 0.0, 0.0);
    slices[0].phase = Vector<3,float>(0.0, 1.0, 0.0);
    slices[0].excitation = new ExcitationPulseSequence(rfcoil);
}

SpinEchoSequence::SpinEchoSequence(PropertyTreeNode* node) 
    : tr(node->GetPath("tr", 0.0f))
    , te(node->GetPath("te", 0.0f))
    , fov(node->GetPath("fov", 0.0f))
{
    unsigned int width  = node->GetPath("samples-x", 0);
    unsigned int height = node->GetPath("samples-y", 0);

    gx    = node->GetPath("max-gx", double(0.0));
    gyMax = node->GetPath("max-gy", double(0.0));

    if (!node->HaveNode("slices"))
        throw Exception("No slices in SpinEchoSequence");

    PropertyTreeNode* slicesNode = node->GetNodePath("slices");
    unsigned int size = slicesNode->GetSize();

    sampler = new CartesianSampler(Vector<3,unsigned int>(width, height, size));

    slices = vector<Slice>(size);            
    for (unsigned int i = 0; i < size; ++i) {
        PropertyTreeNode* slice = slicesNode->GetNodeIdx(i);
        
        slices[i].readout = slice->GetPath("readout-direction", Vector<3,float>());
        slices[i].readout.Normalize();
        Vector<3,float> sliceNorm = slice->GetPath("gradient-direction", Vector<3,float>());
        sliceNorm.Normalize();
        slices[i].phase = slices[i].readout % sliceNorm;
        slices[i].phase.Normalize();

        Quaternion<float> rot(Math::PI * slice->GetPath("rotation-angle", 0.0f) / 180.0f,
                              slice->GetPath("rotation-axis", Vector<3,float>(1.0, 0.0, 0.0)));
        rot.Normalize();
        
        slices[i].readout = rot.RotateVector(slices[i].readout);
        slices[i].phase = rot.RotateVector(slices[i].phase);
        
        slice->SetPath("gradient-direction", rot.RotateVector(sliceNorm));
        slices[i].excitation = new ExcitationPulseSequence(new TestRFCoil(1.0), slice);
    }
}    

SpinEchoSequence::~SpinEchoSequence() {
    delete sampler;
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
    
    double time;
    MRIEvent e;

    sampler->Reset();

    const unsigned int width = sampler->GetDimensions()[0];
    const unsigned int height = sampler->GetDimensions()[1];

    logger.info << "sample width: " << width << logger.end;
    logger.info << "sample height: " << height << logger.end;

    // gyMax = 10e-3; // mT/m
    const double tau = double(height)/(GYRO_HERTZ * gyMax * fov);
    const double gyStart = -gyMax*0.5;
    const double dGy = gyMax / double(height);
    logger.info << "dGY: " << dGy << logger.end;

    // gx = 10e-3; // mT/m
    const double samplingDT = 1.0 / (fov * GYRO_HERTZ * gx);              
    const double gxDuration = samplingDT * double(width);
    logger.info << "sampling dt: " << samplingDT << logger.end;
    const double gxFirst = (gx * gxDuration * 0.5) / tau;


    time = 0.0;
    double start = 0.0;
    for (unsigned int k = 0; k < slices.size(); ++k) {
        for (unsigned int j = 0; j < height; ++j) {
            // time is line * tr + slice * lines * tr
            time = start = double(j) * double(tr) +  double(k) * double(height) * double(tr); 

            unsigned int scanline = height / 2 + (height % 2);
            if (j % 2 == 0) 
                scanline -= j/2 + 1;
            else
                scanline += j/2;

            // e.action = MRIEvent::RESET;
            // seq.push_back(make_pair(time, e));

            // grab and reset the rf sequence for k'th slice
            IMRISequence* pulseSeq = slices[k].excitation;
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
            e.action = MRIEvent::GRADIENT;
            e.gradient = (slices[k].readout * gxFirst) - (slices[k].phase* (gyStart + double(scanline) * dGy));
            seq.push_back(make_pair(time, e));

            // turn off phase and freq encoding gradients
            e.action = MRIEvent::GRADIENT;
            e.gradient = Vector<3,float>(0.0, 0.0, 0.0);
            time += tau;
            seq.push_back(make_pair(time, e));

            //180 degree pulse (cheating!)
            e.action = MRIEvent::INVERT;
            time = start + te * 0.5;
            seq.push_back(make_pair(time, e));
        
            // frequency encoding gradient on
            e.action = MRIEvent::GRADIENT;
            e.gradient = slices[k].readout * gx;
            time = start + te - gxDuration * 0.5;
            seq.push_back(make_pair(time, e));
        
            // record width sample points
            for (unsigned int i = 0; i < width; ++i) {
                e.action = MRIEvent::RECORD;
                e.point = Vector<3,unsigned int>(i, scanline, k);
                seq.push_back(make_pair(time, e));
                time += samplingDT;
            }

            time -= samplingDT;

	    //phase and frequency correction (back to k = (0,0))
	    e.action = MRIEvent::GRADIENT;
	    e.gradient = -(slices[k].readout * gxFirst) - slices[k].phase * (gyStart + double(scanline) * dGy);
	    seq.push_back(make_pair(time, e));

            // frequency encoding gradient off
            e.action = MRIEvent::GRADIENT; 
            e.gradient = Vector<3,float>(0.0);
            seq.push_back(make_pair(time, e));
	    }
    }
    e.action = MRIEvent::DONE;
    seq.push_back(make_pair(time, e));
    
    //Sort();
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

    
    ValueList values2 = ListSequence::Inspect();
    values.merge(values2);
    return values;
}

IMRISampler& SpinEchoSequence::GetSampler() {
    return *sampler;
}


} // NS Science
} // NS MRI
