// MRI simple echo planar echo sequence
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "EchoPlanarSequence.h"

#include "TestRFCoil.h"
#include "ExcitationPulseSequence.h"

#include <Utils/PropertyTreeNode.h>

#include <Math/Quaternion.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils;
using namespace OpenEngine::Math;

EchoPlanarSequence::EchoPlanarSequence(float tr, float te, float fov, Vector<3,unsigned int> dims)
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

EchoPlanarSequence::EchoPlanarSequence(PropertyTreeNode* node) 
    : tr(node->GetPath("tr", 0.0f))
    , te(node->GetPath("te", 0.0f))
    , fov(node->GetPath("fov", 0.0f))
{
    unsigned int width  = node->GetPath("samples-x", 0);
    unsigned int height = node->GetPath("samples-y", 0);

    gx    = node->GetPath("max-gx", double(0.0));
    gyMax = node->GetPath("max-gy", double(0.0));

    if (!node->HaveNode("slices"))
        throw Exception("No slices in EchoPlanarSequence");

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

EchoPlanarSequence::~EchoPlanarSequence() {
    delete sampler;
}

void EchoPlanarSequence::SetFOV(float fov) {
        this->fov = fov;
}

void EchoPlanarSequence::SetTR(float tr) {
        this->tr = 1e-3*tr;
}

void EchoPlanarSequence::SetTE(float te) {
        this->te = 1e-3*te;
}

float EchoPlanarSequence::GetFOV() {
        return fov;
}

float EchoPlanarSequence::GetTR() {
        return tr*1e3;
}
    
float EchoPlanarSequence::GetTE() {
        return te*1e3;
}

void EchoPlanarSequence::Reset(MRISim& sim) {
    ListSequence::Clear();
    
    double time;
    MRIEvent e;

    sampler->Reset();

    const unsigned int width = sampler->GetDimensions()[0];
    const unsigned int height = sampler->GetDimensions()[1];

    logger.info << "sample width: " << width << logger.end;
    logger.info << "sample height: " << height << logger.end;

    // const double gyMax = 10e-3; // mT/m
    const double tau = double(height)/(GYRO_HERTZ * gyMax * fov);
    const double gyStart = -gyMax*0.5;
    const double dGy = gyMax / double(height);
    logger.info << "dGY: " << dGy << logger.end;

    // const double gx = 10e-3; // mT/m
    const double samplingDT = 1.0 / (fov * GYRO_HERTZ * gx);              
    const double gxDuration = samplingDT * double(width);
    logger.info << "sampling dt: " << samplingDT << logger.end;
    const double gxFirst = (gx * gxDuration * 0.5) / tau;


    time = 0.0;
    double start = 0.0;
    for (unsigned int k = 0; k < slices.size(); ++k) {

        time = start = double(k) * double(tr); 

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

        // setup initial phase encoding gradient
        e.action = MRIEvent::GRADIENT;
        e.gradient = (slices[k].readout * gxFirst) + (slices[k].phase * gyStart);
        seq.push_back(make_pair(time, e));

        // turn off phase and freq encoding gradients
        e.action = MRIEvent::GRADIENT;
        e.gradient = Vector<3,float>(0.0, 0.0, 0.0);
        time += tau;
        seq.push_back(make_pair(time, e));

        for (unsigned int j = 0; j < height; ++j) {

            e.action = MRIEvent::GRADIENT; 
            e.gradient = slices[k].phase * dGy;
            time += tau;
            seq.push_back(make_pair(time, e));
            
            // frequency encoding gradient on
            e.action = MRIEvent::GRADIENT;
            e.gradient = slices[k].readout *  ((j % 2 == 0)? -gx : gx); // move back and forth in x
            //Vector<3,float>(-gx, 0.0, 0.0);
            time += tau;
            seq.push_back(make_pair(time, e));
            
            // record width sample points
            for (unsigned int i = 0; i < width; ++i) {
                e.action = MRIEvent::RECORD;
                e.point = Vector<3,unsigned int>(((j % 2 == 0)? i : width - i - 1), j, k);
                seq.push_back(make_pair(time, e));
                time += samplingDT;
            }
            
            time -= samplingDT;
            
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

ValueList EchoPlanarSequence::Inspect() {
    ValueList values;
    
    {
        RWValueCall<EchoPlanarSequence, float > *v
            = new RWValueCall<EchoPlanarSequence, float >(*this,
                                              &EchoPlanarSequence::GetTR,
                                              &EchoPlanarSequence::SetTR);
        v->name = "TR(ms)";
        v->properties[STEP] = 1.0;
        v->properties[MIN]  = 50.0;
        v->properties[MAX]  = 4000.0;
        values.push_back(v);
    }

    {
        RWValueCall<EchoPlanarSequence, float > *v
            = new RWValueCall<EchoPlanarSequence, float >(*this,
                                              &EchoPlanarSequence::GetTE,
                                              &EchoPlanarSequence::SetTE);
        v->name = "TE(ms)";
        v->properties[STEP] = 1.0;
        v->properties[MIN]  = 0.0;
        v->properties[MAX]  = 1000.0;
        values.push_back(v);
    }

    {
        RWValueCall<EchoPlanarSequence, float > *v
            = new RWValueCall<EchoPlanarSequence, float >(*this,
                                              &EchoPlanarSequence::GetFOV,
                                              &EchoPlanarSequence::SetFOV);
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

IMRISampler& EchoPlanarSequence::GetSampler() {
    return *sampler;
}


} // NS Science
} // NS MRI
