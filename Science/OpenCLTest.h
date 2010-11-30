// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_OPEN_C_L_TEST_H_
#define _OE_OPEN_C_L_TEST_H_

#include <Meta/OpenCL.h>

#include <vector>

namespace MRI {
namespace Science {

using std::vector;

/**
 * Short description.
 *
 * @class OpenCLTest OpenCLTest.h s/MRISIM/Science/OpenCLTest.h
 */
class OpenCLTest {
private:
    cl::Context context;
    vector<cl::Device> devices;
    cl::Device currentDevice;
    cl::CommandQueue queue;
    cl::Kernel kernel;

    void InitContext();
    void InitDevice();
    void InitQueue();

    void LoadKernel();
    
public:
    OpenCLTest();



    void RunKernel();
};

} // NS Science
} // NS OpenEngine

#endif // _OE_OPEN_C_L_TEST_H_
