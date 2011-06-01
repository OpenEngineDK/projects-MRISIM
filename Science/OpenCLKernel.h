// MRI Simulator: cpu kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_OPENCL_KERNEL_
#define _MRI_OPENCL_KERNEL_

#include "MRISim.h"
#include "../Resources/Phantom.h"

#include <Meta/OpenCL.h>
#include <Utils/IInspector.h>
#include <Math/RandomGenerator.h>
#include <Scene/RenderNode.h>

// config
#define USE_FLOAT_4 1
#define USE_FAST_REDUCE 1


namespace MRI {
namespace Science {

using Resources::Phantom;
// using MRI::Scene::SpinNode;
using OpenEngine::Math::RandomGenerator;
using OpenEngine::Scene::RenderNode;


struct float2
{
    float data[2];
    float& operator[](unsigned int c)
    {
        return data[c];
    }
};


struct float4
{
    float data[4];
    float& operator[](unsigned int c)
    {
        return data[c];
    }
};


class OpenCLKernel: public IMRIKernel {
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
#if USE_FLOAT_4
    Vector<4,float>* refMagnets;
    Vector<4,float> signal;
#else
    Vector<3,float>* refMagnets;
    Vector<3,float> signal;
#endif
    Vector<3,float>* labMagnets;

    float *eq, *deltaB0;
    Vector<3,float> gradient, rfSignal;
    unsigned char* data;
    unsigned int width, height, depth, sz;
    float b0, gyro;

    
    float2* spinPacks;
    // OpenCL Section
    
    void InitOpenCL();

    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    cl::Program program;

    // Test 4
    cl::Kernel test4kernel;
    cl::Buffer test4buf;
    
    // Step kernel
    cl::Kernel stepKernel;
    cl::Buffer refMBuffer;
    cl::Buffer dataBuffer;
    cl::Buffer spinPackBuffer;
    cl::Buffer eqBuffer;

    // Invert kernel
    cl::Kernel invertKernel;
    
    
    // ---
    
    // Reduce kernel
    cl::Kernel reduceKernel;

    vector<cl::Kernel> reduce_kernels;


    cl::Buffer* inbuffer;
    cl::Buffer* outbuffer;



    cl::Buffer reduceA;
    cl::Buffer reduceB;


    int reduce_count;

    size_t* reduce_group_counts;
    size_t* reduce_item_counts;
    int* reduce_operation_counts;
    int* reduce_entry_counts;
    

    RandomGenerator randomgen;
    
    double time, omega0Angle;

    inline float RandomAttribute(float base, float variance);

    //void SetupTest4();
    void SetupReduceKernel();
    void SetupStepKernel();
    void SetupFastReduceKernel();
    Vector<4, float> FastReduce();
    
    //void Test4();

public:
    OpenCLKernel();
    virtual ~OpenCLKernel();

    void Init(Phantom phantom);
    void Step(float dt);    
    Vector<3,float>* GetMagnets() const;
    Phantom GetPhantom() const;
    // void RFPulse(float angle, unsigned int slice);
    void InvertSpins();
    void Reset();
    void SetGradient(Vector<3,float> gradient);
    void SetRFSignal(Vector<3,float> signal);
    void Flip(unsigned int slice);
    void Flop(unsigned int slice);
    Vector<3,float> GetSignal();
    Vector<3,float> GetGradient() const;

    void SetB0(float b0);
    float GetB0() const;


    //RenderNode *GetRenderNode();

};


} // NS Science
} // NS OpenEngine

#endif // _MRI_OPENCL_KERNEL_
