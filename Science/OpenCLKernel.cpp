// MRI Simulator: cpu kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "OpenCLKernel.h"

#include <Math/Matrix.h>
#include <Logging/Logger.h>
#include <Resources/DirectoryManager.h>

#include <fstream>
#include <iostream>
#include <iterator>


namespace MRI {
namespace Science {

using namespace OpenEngine::Utils::Inspection;
using namespace OpenEngine::Math;
using namespace OpenEngine::Resources;

inline Vector<3,float> RotateX(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(1.0, 0.0, 0.0,
                        0.0, cos(angle), sin(angle),
                        0.0, -sin(angle), cos(angle));
    return m*vec;
}

inline Vector<3,float> RotateY(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(cos(angle), 0.0, sin(angle),
                               0.0, 1.0,        0.0,
                       -sin(angle), 0.0, cos(angle));
    return m*vec;
}

inline Vector<3,float> RotateZ(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(cos(angle), -sin(angle), 0.0,
                        sin(angle), cos(angle), 0.0,
                        0.0, 0.0, 1.0);
    return m*vec;
}


OpenCLKernel::OpenCLKernel() 
    : refMagnets(NULL)
    , labMagnets(NULL)
    , eq(NULL)
    , deltaB0(NULL)
    , gradient(Vector<3,float>(0.0))
    , rfSignal(Vector<3,float>(0.0))
    , data(NULL)
    , width(0)
    , height(0)
    , depth(0)
    , sz(0)
    , b0(0.5)
    , gyro(GYRO_RAD) // radians/Tesla
    , time(0.0)
    , omega0Angle(0.0)
{
    randomgen.SeedWithTime();
    //rn = new KernRenderNode(this);
}

OpenCLKernel::~OpenCLKernel() {
}

float OpenCLKernel::RandomAttribute(float base, float variance) {
    return base + randomgen.UniformFloat(-1.0,1.0) * variance;
}

struct GpuPhantomInfo {
    cl_float4 offset;
    cl_float4 size;
};

void OpenCLKernel::Init(Phantom phantom) {
    this->phantom = phantom;
    width  = phantom.texr->GetWidth();
    height = phantom.texr->GetHeight();
    depth  = phantom.texr->GetDepth();
    sz = width*height*depth;
    refMagnets = new Vector<3,float>[sz]; 
    labMagnets = new Vector<3,float>[sz]; 
    eq = new float[sz];
    deltaB0 = new float[sz];
    data = phantom.texr->GetData();

    for (unsigned int i = 0; i < sz; ++i) {
        //deltaB0[i] = RandomAttribute(0.0, 1e-3);
        deltaB0[i] = 0.0;
        eq[i] = phantom.spinPackets[data[i]].ro*b0;
        // logger.info << "data[" << i << "] = " << int(data[i]) << logger.end;
        // logger.info << "eq[" << i << "] = " << eq[i] << logger.end;
    }

    int sps = phantom.spinPackets.size();
    spinPacks = new cl_float2[sps];
    for (int i=0;i<sps;i++) {
        ((cl_float*)&spinPacks[i])[0] = phantom.spinPackets[i].t1;
        ((cl_float*)&spinPacks[i])[1] = phantom.spinPackets[i].t2;
    }


    string kpath = DirectoryManager::FindFileInPath("Kernels.cl");
    
    
    ifstream file(kpath.c_str());
    string kernelString(istreambuf_iterator<char>(file),
                        (istreambuf_iterator<char>()));

    

    // OpenCL setup, refactor out?
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    logger.warning << "Found " << platforms.size() << " OpenCL platforms!" << logger.end;

    // Let's find a GPU device!
    cl_context_properties properties[] =
        {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    logger.warning << "Found " << devices.size() << " OpenCL devices!" << logger.end;
    
    for (std::vector<cl::Device>::iterator itr = devices.begin();
         itr != devices.end();
         itr++) {
        //logger.info << "Device " << 
    }
    
    device = devices[0];

    cl_int err = CL_SUCCESS;
    queue = new cl::CommandQueue(context, devices[0], 0, &err);
    if (err)
        logger.error << "Created queue: "  << err << logger.end;
    
    const char *kernelStr = kernelString.c_str();

    cl::Program::Sources source(1, std::make_pair(kernelStr, strlen(kernelStr)));
    cl::Program program(context, source);
    //const char *options = "-cl-opt-disable";
    const char *options = NULL;
    err = program.build(devices,options);
    if (err)
        logger.error << "Made program: " << err << logger.end;
    if (err) {
        string log =  program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        logger.error << log << logger.end;
    }
    kernel = new cl::Kernel(program, "mri_step", &err);
    if (err) {
        logger.error << "Made kernel: " << err << logger.end;
        throw Exception("couldn't create kernel");
    }

    reduceKernel = new cl::Kernel(program, "reduce_signal", &err);
    if (err) {
        logger.error << "Made reduce kernel: " << err << logger.end;
        throw Exception("couldn't create kernel");
    }

    refMBuffer = new cl::Buffer(context,
                                CL_MEM_READ_WRITE,
                                sz*sizeof(Vector<3,float>),
                                NULL,
                                &err);

    dataBuffer = new cl::Buffer(context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sz*sizeof(unsigned char),
                                data,
                                &err);

    spinPackBuffer = new cl::Buffer(context,
                                    CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
                                    sps*sizeof(cl_float2),
                                    spinPacks,
                                    &err);
    eqBuffer = new cl::Buffer(context,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sz*sizeof(float),
                              eq,
                              &err);
    
    reduceA = new cl::Buffer(context,
                             CL_MEM_READ_WRITE,
                             sz*sizeof(Vector<3,float>),
                             NULL,
                             &err);
    if (err) {
        logger.error << "Create buffer a: " << err << logger.end;
        throw Exception("couldn't create buffer");
    }
    
    reduceB = new cl::Buffer(context,
                             CL_MEM_READ_WRITE,
                             sz*sizeof(Vector<3,float>),
                             NULL,
                             &err);    
    if (err) {
        logger.error << "Create buffer b: " << err << logger.end;
        throw Exception("couldn't create buffer");

    }
    

    logger.info << "Creating memory " << sz << " " << sz*sizeof(Vector<3,float>) << logger.end;
    inbuffer = new cl::Buffer(context, 
                        CL_MEM_READ_ONLY, 
                        sz*sizeof(Vector<3,float>),
                        NULL,
                        &err);
    if (err) {
        logger.error << "creating memory = " << err << logger.end;
        throw Exception("couldn't create buffer");
    }

    outbuffer = new cl::Buffer(context,
                         CL_MEM_WRITE_ONLY,
                         sz*sizeof(Vector<3,float>),
                         NULL,
                         &err);
    
    kernel->setArg(1, *refMBuffer);
    kernel->setArg(2, *dataBuffer);
    kernel->setArg(3, *spinPackBuffer);
    kernel->setArg(4, *eqBuffer);

    GpuPhantomInfo pinfo;
    

    ((cl_float*)&pinfo.offset)[0] = phantom.offsetX;;
    ((cl_float*)&pinfo.offset)[1] = phantom.offsetY;;
    ((cl_float*)&pinfo.offset)[2] = phantom.offsetZ;;
    ((cl_float*)&pinfo.offset)[3] = 0;

    ((cl_float*)&pinfo.size)[0] = phantom.sizeX;
    ((cl_float*)&pinfo.size)[1] = phantom.sizeY;
    ((cl_float*)&pinfo.size)[2] = phantom.sizeZ;
    ((cl_float*)&pinfo.size)[3] = 0.0;

    //kernel->setArg(6, sizeof(GpuPhantomInfo), &pinfo);
    err = kernel->setArg(6, sizeof(cl_float4), &pinfo.offset);

    if (err) {
        logger.error << "setting args = " << err << logger.end;
        throw Exception("couldn't set args");
    }
    

    err = kernel->setArg(7, sizeof(cl_float4), &pinfo.size);


    //kernel->setArg(0, *inbuffer);
    //kernel->setArg(1, *outbuffer);
    if (err) {
        logger.error << "setting args = " << err << logger.end;
        throw Exception("couldn't set args");
    }

    SetupReduce();

    Reset();

}



void OpenCLKernel::Step(float dt) {
    

    time += dt;
    signal = Vector<3,float>();
    const double omega0 = GYRO_RAD * b0;
    omega0Angle += dt * omega0;
    if (omega0Angle > double(Math::PI * 2.0))
        omega0Angle = fmod(omega0Angle, double(Math::PI * 2.0));

    // move rf signal into reference space
    const Vector<3,float> rf = RotateZ(-omega0Angle, rfSignal);

    cl::Event event;

    cl_float2 rf_cl;
    ((cl_float*)&rf_cl)[0] = rf.Get(0);
    ((cl_float*)&rf_cl)[1] = rf.Get(1);

    kernel->setArg(8, sizeof(cl_float2), &rf_cl);

    cl_float4 vec;
    gradient.ToArray(((cl_float*)&vec));
    
    kernel->setArg(5, 4*sizeof(float), ((cl_float*)&vec));

    kernel->setArg(0,dt);
    cl_int err = queue->enqueueNDRangeKernel(*kernel, 
                                             cl::NullRange,
                                             cl::NDRange(width,height,depth),
                                             cl::NullRange,
                                             //cl::NDRange(1,1,1),
                                             NULL,                                             
                                             &event);


    

    if (err) {
        logger.error << "error "  << err << logger.end;
        throw Exception("couldn't run kernel");

    }
        
    //queue->enqueueReadBuffer(*refMBuffer, CL_TRUE, 0, sz*sizeof(Vector<3,float>), refMagnets, NULL, &event);
    event.wait();


    

}

void OpenCLKernel::Flip(unsigned int slice) {
    RFPulse(Math::PI * 0.5, slice);
}

void OpenCLKernel::Flop(unsigned int slice) {
    RFPulse(Math::PI, slice);
}


void create_reduction_pass_counts(
    int count, 
    int max_group_size,    
    int max_groups,
    int max_work_items, 
    int *pass_count, 
    size_t **group_counts, 
    size_t **work_item_counts,
    int **operation_counts,
    int **entry_counts)
{
    int work_items = (count < max_work_items * 2) ? count / 2 : max_work_items;
    if(count < 1)
        work_items = 1;
        
    int groups = count / (work_items * 2);
    groups = max_groups < groups ? max_groups : groups;

    int max_levels = 1;
    int s = groups;

    while(s > 1) 
    {
        int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        s = s / (work_items*2);
        max_levels++;
    }
 
    *group_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    *work_item_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    *operation_counts = (int*)malloc(max_levels * sizeof(int));
    *entry_counts = (int*)malloc(max_levels * sizeof(int));

    (*pass_count) = max_levels;
    (*group_counts)[0] = groups;
    (*work_item_counts)[0] = work_items;
    (*operation_counts)[0] = 1;
    (*entry_counts)[0] = count;
    if(max_group_size < work_items)
    {
        (*operation_counts)[0] = work_items;
        (*work_item_counts)[0] = max_group_size;
    }
    
    s = groups;
    int level = 1;
   
    while(s > 1) 
    {
        int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        int groups = s / (work_items * 2);
        groups = (max_groups < groups) ? max_groups : groups;

        (*group_counts)[level] = groups;
        (*work_item_counts)[level] = work_items;
        (*operation_counts)[level] = 1;
        (*entry_counts)[level] = s;
        if(max_group_size < work_items)
        {
            (*operation_counts)[level] = work_items;
            (*work_item_counts)[level] = max_group_size;
        }
        
        s = s / (work_items*2);
        level++;
    }
}


#define MAX_GROUPS      (64)
#define MAX_WORK_ITEMS  (64)


void OpenCLKernel::SetupReduce() {

    int count = sz;
    size_t max_workgroup_size = 0;

    int pass_count = 0;
    size_t* group_counts = 0;
    size_t*          work_item_counts = 0;
    int*             operation_counts = 0;
    int*             entry_counts = 0;

    max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    // int err = clGetDeviceInfo(device_id,
    //                           CL_DEVICE_MAX_WORK_GROUP_SIZE, 
    //                           sizeof(size_t),
    //                           &max_workgroup_size,
    //                           &returned_size);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to retrieve device info!\n");
    //     return EXIT_FAILURE;
    // }

    create_reduction_pass_counts(
        count, max_workgroup_size, 
        MAX_GROUPS, MAX_WORK_ITEMS, 
        &pass_count, &group_counts, 
        &work_item_counts, &operation_counts,
        &entry_counts);

    logger.error << pass_count << logger.end;

}

void OpenCLKernel::Reduce() const {
    
}

Vector<3,float> OpenCLKernel::GetSignal() const {
    // reduce!

    Reduce();

    cl::Event event;
    int size = width*height*depth;
    int i =0;
    cl::Buffer* lastBuf;
    while (size > 1) {
        if (i % 2) {
            reduceKernel->setArg(0, *reduceA);
            reduceKernel->setArg(1, *reduceB);
            lastBuf = reduceB;
        } else {
            reduceKernel->setArg(0, *reduceB);
            reduceKernel->setArg(1, *reduceA);
            lastBuf = reduceA;
        }        

        if (i == 0) {
            reduceKernel->setArg(0, *refMBuffer);
        }

        size = ceil(size/2.0);


        cl_int err = queue->enqueueNDRangeKernel(*reduceKernel,
                                                 cl::NullRange,
                                                 cl::NDRange(size),
                                                 cl::NullRange,
                                                 NULL,
                                                 &event);
        if (err) {
            logger.error << "error = " << err << logger.end;
            throw Exception("couldn't run reduce");                    
        }
        i++;
    }

    Vector<3,float> signal2;
    queue->enqueueReadBuffer(*lastBuf, CL_TRUE, 0, sizeof(Vector<3,float>), &signal2, NULL, &event);
    event.wait();
    //logger.warning << signal2 << logger.end;

    //signal = signal2;


    return signal2;
}

void OpenCLKernel::SetB0(float b0) {
    this->b0 = b0;
}

float OpenCLKernel::GetB0() const {
    return b0;
}

void OpenCLKernel::RFPulse(float angle, unsigned int slice) {
    Matrix<3,3,float> rot(
                          1.0, 0.0, 0.0,
                          0.0, cos(angle), sin(angle),
                          0.0,-sin(angle), cos(angle)
                          );
    // hack to only excite slice 0
    unsigned int z = slice;
    for (unsigned int i = 0; i < width; ++i) {
        for (unsigned int j = 0; j < height; ++j) {
            if (data[i + j*width  + z*width*height] == 0) continue;
            refMagnets[i + j*width + z*width*height] = rot*refMagnets[i + j*width + z*width*height];
        }
    }
}

void OpenCLKernel::SetGradient(Vector<3,float> gradient) {
    this->gradient = gradient;
}

Vector<3,float> OpenCLKernel::GetGradient() const {
    return gradient;
}

void OpenCLKernel::SetRFSignal(Vector<3,float> signal) {
    rfSignal = signal;
}

void OpenCLKernel::Reset() {
    // initialize refMagnets to b0 * spin density 
    // Signal should at all times be the sum of the spins (or not?)
    omega0Angle = time = 0.0;
    signal = Vector<3,float>();
    for (unsigned int i = 0; i < sz; ++i) {
        refMagnets[i] = labMagnets[i] = Vector<3,float>(0.0, 0.0, eq[i]);
        signal += labMagnets[i];
        //signal += refMagnets[i];
    }
    cl::Event event;
    queue->enqueueWriteBuffer(*refMBuffer, CL_TRUE, 0, sz*sizeof(Vector<3, float>), refMagnets, NULL, &event);
    event.wait();


    logger.info << "Signal: " << signal << logger.end;
}

Vector<3,float>* OpenCLKernel::GetMagnets() const {
    //return refMagnets;
    return labMagnets;
}

Phantom OpenCLKernel::GetPhantom() const {
  return phantom;
}

// RenderNode* OpenCLKernel::GetRenderNode() {
//     return rn;
// }

// OpenCLKernel::KernRenderNode::KernRenderNode(OpenCLKernel *k) : kern(k) {

// }

// void OpenCLKernel::KernRenderNode::Apply(Renderers::RenderingEventArg arg, OpenEngine::Scene::ISceneNodeVisitor& v) {

//     using namespace OpenEngine::Renderers;

//     float scale = 10;

//     IRenderer& rend = arg.renderer;
//     Vector<3,float> zero(0,0,0);
    
//     Line xaxis(zero, Vector<3,float>(1,0,0)*scale);
//     Line yaxis(zero, Vector<3,float>(0,1,0)*scale);
//     Line zaxis(zero, Vector<3,float>(0,0,1)*scale);

//     rend.DrawLine(xaxis, Vector<3,float>(1,0,0));
//     rend.DrawLine(yaxis, Vector<3,float>(0,1,0));
//     rend.DrawLine(zaxis, Vector<3,float>(0,0,1));

//     Line l(zero, magnet*scale);
//     Line l2(magnet*scale, (magnet*scale+rf*400));
    
//     rend.DrawLine(l, Vector<3,float>(1,0,0),2);
//     rend.DrawLine(l2, Vector<3,float>(0,1,0),2);
    


// }

} // NS Science
} // NS OpenEngine
