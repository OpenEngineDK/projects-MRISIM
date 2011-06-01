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
    float4 offset;
    float4 size;
};

void CLErrorCheck(cl_int e, string msg) {
    if (e != CL_SUCCESS) {
        logger.error << "[OpenCL] (" << e << ") " << msg << logger.end;
        throw Exception(msg);
    }
}

void context_err_callback (const char *errinfo, 
                           const void *private_info, 
                           size_t cb, 
                           void *user_data) {
    logger.error << "[OpenCL] callback: " << errinfo << logger.end;
}
                  

void OpenCLKernel::InitOpenCL() {
    cl_int err;

    // Fidn platforms
    vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        logger.error << "No OpenCL platforms found" << logger.end;
        throw Exception("No OpenCL platforms found"); 
    }
    logger.info << "[OpenCL] found " << platforms.size() << " platforms" << logger.end;
    cl::Platform plat = platforms[0];
    
    // Find devices
    vector<cl::Device> devices;
    err = plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    CLErrorCheck(err, "Unable to get devices");
    if (devices.size() == 0) {
        logger.error << "[OpenCL] No devices found" << logger.end;
        throw Exception("no opencl devices");
    }
    logger.info << "[OpenCL] found " << devices.size() << " devices" << logger.end;
    device = devices[0];

    // Create context
    context = cl::Context(devices, NULL, NULL, NULL, &err);
    CLErrorCheck(err, "Unable to create context");

    
    // Create queue
    queue = cl::CommandQueue(context, device, 0, &err);
    CLErrorCheck(err, "Unable to create command queue");
        
    // Load kernel source
    string kpath = DirectoryManager::FindFileInPath("Kernels.cl");
    ifstream file(kpath.c_str());
    string kernelString(istreambuf_iterator<char>(file),
                        (istreambuf_iterator<char>()));
    const char *kernelStr = kernelString.c_str();
    cl::Program::Sources source(1, std::make_pair(kernelStr, strlen(kernelStr)));
    
    // Create program
    program = cl::Program(context, source, &err);
    CLErrorCheck(err, "Unable to create program");
    
    // Compile!
    const char *options = "";
    
    err = program.build(devices, options);
    if (err != CL_SUCCESS) {
        string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        logger.error << "[OpenCL] Compile failed: " << log << logger.end;
        throw Exception("Compile error");
    }

    // Find kernels
    cl_int kerr;
    
#if USE_FLOAT_4
    stepKernel = cl::Kernel(program, "mri_step4", &kerr); err |= kerr;    
    reduceKernel = cl::Kernel(program, "reduce_signal4", &kerr); err |= kerr;
    invertKernel = cl::Kernel(program, "invert_kernel4", &kerr); err |= kerr;
#else
    stepKernel = cl::Kernel(program, "mri_step", &kerr); err |= kerr;
    reduceKernel = cl::Kernel(program, "reduce_signal", &kerr); err |= kerr;
    invertKernel = cl::Kernel(program, "invert_kernel", &kerr); err |= kerr;
#endif

        
    CLErrorCheck(err, "Unable to get kernels");
    
    // Extra kernels setup
    
    SetupStepKernel();

    SetupReduceKernel();
#if USE_FAST_REDUCE
    SetupFastReduceKernel();
#endif


    
    // setup invert kernel
    invertKernel.setArg(0, refMBuffer);


}

void OpenCLKernel::Init(Phantom phantom) {
    this->phantom = phantom;
    width  = phantom.texr->GetWidth();
    height = phantom.texr->GetHeight();
    depth  = phantom.texr->GetDepth();
    sz = width*height*depth;
#if USE_FLOAT_4
    refMagnets = new Vector<4,float>[sz]; 
#else
    refMagnets = new Vector<3,float>[sz]; 
#endif

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
    spinPacks = new float2[sps];
    for (int i=0;i<sps;i++) {
        ((cl_float*)&spinPacks[i])[0] = phantom.spinPackets[i].t1;
        ((cl_float*)&spinPacks[i])[1] = phantom.spinPackets[i].t2;
    }



    InitOpenCL();


    Reset();


}



void OpenCLKernel::Step(float dt) {
    time += dt;
#if USE_FLOAT_4
    signal = Vector<4,float>();
#else
    signal = Vector<3,float>();
#endif
    const double omega0 = GYRO_RAD * b0;
    omega0Angle += dt * omega0;
    if (omega0Angle > double(Math::PI * 2.0))
        omega0Angle = fmod(omega0Angle, double(Math::PI * 2.0));

    // move rf signal into reference space
    //    const Vector<3,float> rf = RotateZ(-omega0Angle, rfSignal);
    const Vector<3,float> rf = rfSignal;

    cl::Event event;

    cl_float2 rf_cl;
    ((cl_float*)&rf_cl)[0] = rf.Get(0);
    ((cl_float*)&rf_cl)[1] = rf.Get(1);

    stepKernel.setArg(8, sizeof(cl_float2), &rf_cl);

    cl_float4 vec;
    gradient.ToArray(((cl_float*)&vec));
    
    stepKernel.setArg(5, 4*sizeof(float), ((cl_float*)&vec));

    stepKernel.setArg(0,dt);
    cl_int err = queue.enqueueNDRangeKernel(stepKernel, 
                                            cl::NullRange,
                                            //cl::NDRange(sz),
                                            cl::NDRange(width,height,depth),
                                            cl::NullRange,
                                            //cl::NDRange(1,1,1),
                                            NULL,                                             
                                            &event);


    

    if (err) {
        logger.error << "error "  << err << logger.end;
        throw Exception("couldn't run kernel");

    }

    event.wait();        

}

void OpenCLKernel::Flip(unsigned int slice) {
    // RFPulse(Math::PI * 0.5, slice);
}

void OpenCLKernel::Flop(unsigned int slice) {
    // RFPulse(Math::PI, slice);
}




#define MAX_GROUPS      (64)
#define MAX_WORK_ITEMS  (64)


void OpenCLKernel::SetupFastReduceKernel() {
    
    unsigned int count = sz;
    size_t max_group_size = 0;

    max_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

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

    string kpath = DirectoryManager::FindFileInPath("reduce_float4_kernel.cl");    
    
    ifstream file(kpath.c_str());
    string kernelString(istreambuf_iterator<char>(file),
                        (istreambuf_iterator<char>()));
    const char* source = kernelString.c_str();

    // create_reduction_pass_counts(
    //     count, max_workgroup_size, 
    //     MAX_GROUPS, MAX_WORK_ITEMS, 
    //     &reduce_count, &group_counts, 
    //     &work_item_counts, &operation_counts,
    //     &entry_counts);


    unsigned int max_work_items = MAX_WORK_ITEMS;
    unsigned int max_groups = MAX_GROUPS;

    /// Create reduction pass counts
    unsigned int work_items = (count < max_work_items * 2) ? count / 2 : max_work_items;
    if(count < 1)
        work_items = 1;
        
    unsigned int groups = count / (work_items * 2);
    groups = max_groups < groups ? max_groups : groups;

    unsigned int max_levels = 1;
    unsigned int s = groups;

    while(s > 1) 
    {
        int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        s = s / (work_items*2);
        max_levels++;
    }
 
    reduce_group_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    reduce_item_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    reduce_operation_counts = (int*)malloc(max_levels * sizeof(int));
    reduce_entry_counts = (int*)malloc(max_levels * sizeof(int));

    reduce_count = max_levels;
    (reduce_group_counts)[0] = groups;
    (reduce_item_counts)[0] = work_items;
    (reduce_operation_counts)[0] = 1;
    (reduce_entry_counts)[0] = count;
    if(max_group_size < work_items)
    {
        (reduce_operation_counts)[0] = work_items;
        (reduce_item_counts)[0] = max_group_size;
    }
    
    s = groups;
    int level = 1;
   
    while(s > 1) 
    {
        unsigned int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        unsigned int groups = s / (work_items * 2);
        groups = (max_groups < groups) ? max_groups : groups;

        (reduce_group_counts)[level] = groups;
        (reduce_item_counts)[level] = work_items;
        (reduce_operation_counts)[level] = 1;
        (reduce_entry_counts)[level] = s;
        if(max_group_size < work_items)
        {
            (reduce_operation_counts)[level] = work_items;
            (reduce_item_counts)[level] = max_group_size;
        }
        
        s = s / (work_items*2);
        level++;
    }


    // done

    vector<cl::Program> programs(reduce_count); // = vector<cl::Program>();

    reduce_kernels = vector<cl::Kernel>(reduce_count);
    

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    cl_int err;

    for (int i=0; i<reduce_count; i++) {
        char *block_source = (char*)malloc(strlen(source) + 1024);
        size_t source_length = strlen(source) + 1024;
        memset(block_source, 0, source_length);
        
        // Insert macro definitions to specialize the kernel to a particular group size
        //
        const char group_size_macro[] = "#define GROUP_SIZE";
        const char operations_macro[] = "#define OPERATIONS";
        sprintf(block_source, "%s (%d) \n%s (%d)\n\n%s\n", 
            group_size_macro, (int)reduce_group_counts[i], 
            operations_macro, (int)reduce_operation_counts[i], 
                source);

        cl::Program::Sources source(1, std::make_pair(block_source, strlen(block_source)));
        programs[i] = cl::Program(context, source);
        //const char *options = "-cl-opt-disable";
        const char *options = NULL;
        err = programs[i].build(devices,options);

        if (err) {
            string log =  programs[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            logger.error << log << logger.end;
            throw Exception("OpenCL complie error");
        }
        reduce_kernels[i] = cl::Kernel(programs[i], "reduce", &err);
        CLErrorCheck(err, "could not create fast reduce kernel");


    }
    logger.info << "Fast Reduce ready" << logger.end;
}

Vector<4, float> OpenCLKernel::FastReduce() {

    logger.error << "reducing..." << logger.end;
    size_t typesize = sizeof(float) ; // more to static?
    int channels = 4;
    cl_int err;

    cl::Buffer* lastBuf;

    for (int i=0; i<reduce_count; i++) {        
        size_t global = reduce_group_counts[i] * reduce_item_counts[i];        
        size_t local = reduce_item_counts[i];
        unsigned int operations = reduce_operation_counts[i];
        unsigned int entries = reduce_entry_counts[i];
        size_t shared_size = typesize * channels * local * operations;
   
            

        err = CL_SUCCESS;
        cl::Kernel kern = reduce_kernels[i];
        
                
        if (i % 2) {
            err |= kern.setArg(0, reduceB);
            err |= kern.setArg(1, reduceA);
            lastBuf = &reduceB;
        } else {
            err |= kern.setArg(0, reduceA);
            err |= kern.setArg(1, reduceB);
            lastBuf = &reduceA;
        }        
        
        if (i == 0) {
            err |= kern.setArg(1, refMBuffer);
        }
        
        err |= kern.setArg(2, cl::__local(shared_size));
        err |= kern.setArg(3, entries);
        CLErrorCheck(err, "Settings args");
        
        err = queue.enqueueNDRangeKernel(kern, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));                        
        CLErrorCheck(err, "fast reduce error");
    }
//    err = queue.finish();
//    CLErrorCheck(err, "fast reduce finish error");
    Vector<4,float> result;
    err = queue.enqueueReadBuffer(*lastBuf, CL_TRUE, 0, sizeof(Vector<4,float>), &result);
    CLErrorCheck(err, "read result");
    logger.error << result << logger.end;
    return result;
    
}
void OpenCLKernel::SetupReduceKernel() {
    cl_int err = 0,berr;
#if USE_FLOAT_4
    size_t size = sz*sizeof(float4);
#else
    size_t size = sz*sizeof(Vector<3,float>);
#endif
    reduceA = cl::Buffer(context,
                         CL_MEM_READ_WRITE,
                         size,
                         NULL,
                         &berr);    
    err |= berr;
    reduceB = cl::Buffer(context,
                         CL_MEM_READ_WRITE,
                         size,
                         NULL,
                         &err);    
    err |= berr;
    CLErrorCheck(err, "Unable to create reduce buffers");
}
    
void OpenCLKernel::SetupStepKernel() {
    
    cl_int err = 0, berr;
#if USE_FLOAT_4
    refMBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, 
                            sz*sizeof(float4),NULL,&berr); err |= berr;
#else
    refMBuffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                            sz*sizeof(Vector<3, float>),NULL,&berr); err |= berr;
#endif
    dataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sz*sizeof(unsigned char), data, &berr); err |= berr;
    int sps = phantom.spinPackets.size();
    
    
    spinPackBuffer = cl::Buffer(context,
                                    CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, sps*sizeof(cl_float2),
                                    spinPacks, &berr); err |= berr;
    eqBuffer = cl::Buffer(context,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz*sizeof(float),
                              eq, &berr); err |= berr;
  
    CLErrorCheck(err, "Unable to create buffers");
    
    err |= stepKernel.setArg(1, refMBuffer);
    err |= stepKernel.setArg(2, dataBuffer);
    err |= stepKernel.setArg(3, spinPackBuffer);
    err |= stepKernel.setArg(4, eqBuffer);
  
    GpuPhantomInfo pinfo;
    
    
    pinfo.offset[0] = phantom.offsetX;;
    pinfo.offset[1] = phantom.offsetY;;
    pinfo.offset[2] = phantom.offsetZ;;
    pinfo.offset[3] = 0;
    
    pinfo.size[0] = phantom.sizeX;
    pinfo.size[1] = phantom.sizeY;
    pinfo.size[2] = phantom.sizeZ;
    pinfo.size[3] = 0.0;
    
    
    err |= stepKernel.setArg(6, sizeof(cl_float4), &pinfo.offset);
    err |= stepKernel.setArg(7, sizeof(cl_float4), &pinfo.size);

    CLErrorCheck(err, "Unable to set args");
    
}

Vector<3,float> OpenCLKernel::GetSignal() {
    
    // reduce!
#if USE_FAST_REDUCE
    Vector<3, float> signal3;
    Vector<4,float> signal4 = FastReduce();
    signal3[0] = signal4[0];
    signal3[1] = signal4[1];
    signal3[2] = signal4[2];
    return signal3;

#else

    cl::Event event;
    int size = width*height*depth;
    int i =0;
    cl::Buffer* lastBuf;
    while (size > 1) {
        if (i % 2) {
            reduceKernel.setArg(0, reduceA);
            reduceKernel.setArg(1, reduceB);
            lastBuf = &reduceB;
        } else {
            reduceKernel.setArg(0, reduceB);
            reduceKernel.setArg(1, reduceA);
            lastBuf = &reduceA;
        }        

        if (i == 0) {
            reduceKernel.setArg(0, refMBuffer);
        }

        size = ceil(size/2.0);
        
        
        cl_int err = queue.enqueueNDRangeKernel(reduceKernel,
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
#if USE_FLOAT_4
    Vector<3,float> signal3;
    Vector<4,float> signal4;
    queue.enqueueReadBuffer(*lastBuf, CL_TRUE, 0,
                             sizeof(float4), &signal4, NULL, &event);
    event.wait();

    signal3[0] = signal4[0];
    signal3[1] = signal4[1];
    signal3[2] = signal4[2];
#else
    Vector<3,float> signal3;
    queue.enqueueReadBuffer(*lastBuf, CL_TRUE, 0, sizeof(Vector<3,float>), &signal3, NULL, &event);
    event.wait();
#endif

    //logger.warning << signal3 << logger.end;

    //signal = signal2;

    return signal3;
#endif
    
}

void OpenCLKernel::SetB0(float b0) {
    this->b0 = b0;
}

float OpenCLKernel::GetB0() const {
    return b0;
}

// void OpenCLKernel::RFPulse(float angle, unsigned int slice) {
//     Matrix<3,3,float> rot(
//                           1.0, 0.0, 0.0,
//                           0.0, cos(angle), sin(angle),
//                           0.0,-sin(angle), cos(angle)
//                           );
//     // hack to only excite slice 0
//     unsigned int z = slice;
//     for (unsigned int i = 0; i < width; ++i) {
//         for (unsigned int j = 0; j < height; ++j) {
//             if (data[i + j*width  + z*width*height] == 0) continue;
//             refMagnets[i + j*width + z*width*height] = rot*refMagnets[i + j*width + z*width*height];
//         }
//     }
// }

void OpenCLKernel::InvertSpins() {    
    return;
    cl::Event event;    
    cl_int err = queue.enqueueNDRangeKernel(invertKernel, 
                                            cl::NullRange,
                                            cl::NDRange(sz),
                                            cl::NullRange,
                                            NULL,                                             
                                            &event);

    if (err) {
        logger.error << "error "  << err << logger.end;
        throw Exception("couldn't run kernel");
        
    }    
    event.wait();
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
#if USE_FLOAT_4
    signal = Vector<4,float>();
#else
    signal = Vector<3,float>();
#endif


    for (unsigned int i = 0; i < sz; ++i) {
#if USE_FLOAT_4
        refMagnets[i] = Vector<4,float>(0.0, 0.0, eq[i], 0.0);
#else
        refMagnets[i] = Vector<3,float>(0.0, 0.0, eq[i]);
#endif
        //signal += labMagnets[i];
        signal += refMagnets[i];
    }
    cl::Event event;
#if USE_FLOAT_4
    queue.enqueueWriteBuffer(refMBuffer, CL_TRUE, 0, sz*sizeof(cl_float4), refMagnets, NULL, &event);
#else
    queue.enqueueWriteBuffer(refMBuffer, CL_TRUE, 0, sz*sizeof(Vector<3, float>), refMagnets, NULL, &event);
#endif
    event.wait();


    //logger.info << "Signal: " << signal << logger.end;
    
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
