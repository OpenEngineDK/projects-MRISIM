#include <Meta/OpenCL.h>

#include "OpenCLTest.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <Logging/Logger.h>

#include <Core/Exceptions.h>

namespace MRI {
namespace Science {

using namespace std;
using namespace OpenEngine::Core;

OpenCLTest::OpenCLTest() {
    InitContext();
    InitDevice();
    InitQueue();

    LoadKernel();
}

void OpenCLTest::InitContext() {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    string pname = platforms[0].getInfo<CL_PLATFORM_NAME>();
    logger.error << "Platfom: " << pname << logger.end;


    cl_context_properties properties[] = 
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    context = cl::Context(CL_DEVICE_TYPE_CPU, properties);

}
void OpenCLTest::InitDevice() {
    devices = context.getInfo<CL_CONTEXT_DEVICES>();    
    currentDevice = devices[0];
    logger.error << currentDevice.getInfo<CL_DEVICE_NAME>() << logger.end;

}

void OpenCLTest::InitQueue() {
    cl_int err = CL_SUCCESS;

    queue = cl::CommandQueue(context, currentDevice, 0, &err);
    logger.error << "Created queue: " << err << logger.end;
}


void OpenCLTest::LoadKernel() {
    cl_int err = CL_SUCCESS;

    ifstream file("projects/MRISIM/test.cl");
    if (file.fail()) {
        throw Exception("Source not found");
    }
    string source(istreambuf_iterator<char>(file),(istreambuf_iterator<char>()));
    
    cl::Program::Sources sourceList(1, make_pair(source.c_str(), source.length()));
    cl::Program program = cl::Program(context, sourceList);
    err = program.build(devices);
    
    kernel = cl::Kernel(program, "test1", &err);    
}

void OpenCLTest::RunKernel() {
    cl_int err;
    cl::Event kevent;
    cl::Event inevent;
    cl::Event outevent;

    float indat[] = {4,3,2,1};
    float outdat[4];
    cl::Buffer input(context, 
                     CL_MEM_READ_ONLY, 
                     sizeof(float)*4);
    cl::Buffer output(context,
                      CL_MEM_WRITE_ONLY,
                      sizeof(float)*4);

    kernel.setArg(0,input);
    kernel.setArg(1,output);

    queue.enqueueWriteBuffer(input,
                             CL_FALSE,
                             0,
                             sizeof(float)*4,
                             indat,
                             NULL,
                             &inevent);
    vector<cl::Event> inevs(1,inevent);
    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(4),
                               cl::NullRange,
                               &inevs,
                               &kevent);
    vector<cl::Event> kes(1,kevent);
    queue.enqueueReadBuffer(output,
                            CL_FALSE,
                            0,
                            sizeof(float)*4,
                            outdat,
                            &kes,
                            &outevent);
    

    err = outevent.wait();
    logger.error << "kernel done: " << err << logger.end;
    for (int i=0;i<4;i++) {
        logger.error << " " << outdat[i]; 
    }
    logger.error << logger.end;
}

}
}
