// MRI Simulator: cpu kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_CPU_KERNEL_
#define _MRI_CPU_KERNEL_

#include "MRISim.h"
#include "../Resources/Phantom.h"

#include <Utils/IInspector.h>
#include <Math/RandomGenerator.h>
#include <Scene/RenderNode.h>

namespace MRI {
namespace Science {

using Resources::Phantom;
using OpenEngine::Math::RandomGenerator;
using OpenEngine::Scene::RenderNode;

class CPUKernel: public IMRIKernel {
private:

    // class KernRenderNode : public RenderNode {
    //     CPUKernel *kern;
    // protected:
    //     friend class CPUKernel;
    //     Vector<3,float> magnet;
    //     Vector<3,float> rf;
    // public:
    //     KernRenderNode(CPUKernel* k);
    //     void Apply(Renderers::RenderingEventArg arg, OpenEngine::Scene::ISceneNodeVisitor& v);
    // };

    // KernRenderNode *rn;

    Phantom phantom;
    Vector<3,float>* refMagnets, *labMagnets;
    float *eq, *deltaB0;
    Vector<3,float> gradient, rfSignal;
    unsigned char* data;
    unsigned int width, height, depth, sz;
    float b0, gyro;
    Vector<3,float> signal;
    RandomGenerator randomgen;
    
    double time, omega0Angle;

    inline float RandomAttribute(float base, float variance);
public:
    CPUKernel();
    virtual ~CPUKernel();

    void Init(Phantom phantom);
    void Step(float dt);    
    Vector<3,float>* GetMagnets() const;
    Phantom GetPhantom() const;
    void RFPulse(float angle, unsigned int slice);
    void Reset();
    void SetGradient(Vector<3,float> gradient);
    void SetRFSignal(Vector<3,float> signal);
    void Flip(unsigned int slice);
    void Flop(unsigned int slice);
    Vector<3,float> GetSignal() const;
    Vector<3,float> GetGradient() const;

    void SetB0(float b0);
    float GetB0() const;

    // RenderNode *GetRenderNode();

};


} // NS Science
} // NS OpenEngine

#endif // _MRI_CPU_KERNEL_
