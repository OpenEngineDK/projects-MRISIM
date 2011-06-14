// MRI command line parser
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "MRICommandLine.h"

#include "Science/MRISim.h"
#include "Science/CPUKernel.h"
#if ENABLE_OPENCL
#include "Science/OpenCLKernel.h"
#endif
#if ENABLE_CUDA
#include "Science/CUDAKernel.h"
#endif
#include "Science/SpinEchoSequence.h"
#include "Science/GradientEchoSequence.h"
#include "Science/EchoPlanarSequence.h"
#include "Science/ExcitationPulseSequence.h"
#include "Science/TestRFCoil.h"
#include "Resources/SimplePhantomBuilder.h"
#include "Resources/TestPhantomBuilder.h"

#include <Utils/PropertyTree.h>
#include <Utils/PropertyTreeNode.h>

using namespace MRI::Resources;
using namespace MRI::Science;

MRICommandLine::MRICommandLine(int argc, char* argv[])
    : sequence(NULL)
    , kernel(NULL)
{
    bool useCPU = false;
#if ENABLE_CUDA
    bool useCUDA = false;
#endif
    string yamlSequence, yamlPhantom;
    unsigned int phantomW = 20;
    unsigned int phantomH = 20;
    unsigned int phantomD = 20;
    
    for (int i=1;i<argc;i++) {
        if (strcmp(argv[i],"-cpu") == 0)
            useCPU = true;
#if ENABLE_CUDA
        else if (strcmp(argv[i],"-cuda") == 0) {
            useCUDA = true;
        }
#endif
        else if (strcmp(argv[i],"-phantom") == 0) {
            if (i + 1 < argc)
                yamlPhantom = string(argv[i+1]);
        }
        else if (strcmp(argv[i],"-sequence") == 0) {
            if (i + 1 < argc)
                yamlSequence = string(argv[i+1]);
        }
        else if (strcmp(argv[i],"-dims") == 0) {
            if (i + 3 < argc) {
                phantomW = strtol(argv[i+1], NULL, 10);
                phantomH = strtol(argv[i+2], NULL, 10);
                phantomD = strtol(argv[i+3], NULL, 10);
            }
        }
        else {
            // unsigned int f = strtol(argv[i], NULL, 10);
            // if (f > 0)
            //     phantomW = phantomH = phantomD = f;
        }
    }

    // load kernel
    if (useCPU)
        kernel = new CPUKernel();
#if ENABLE_CUDA
    else if (useCUDA)
        kernel = new CUDAKernel();
#endif
    else
#if ENABLE_OPENCL
        kernel = new OpenCLKernel();
#else
        kernel = new CPUKernel();
#endif

    // load phantom
    if (yamlPhantom.empty()) {
        IPhantomBuilder* pb = new TestPhantomBuilder(Vector<3,unsigned int>(phantomW, phantomH, phantomD));
        phantom = pb->GetPhantom();
    }
    else {
        phantom = Phantom(yamlPhantom);
    }

    // load sequence
    if (yamlSequence.empty()) {
        sequence = new SpinEchoSequence(2500.0f, 50.0f, phantom.sizeX * 1e-3 * float(phantom.texr->GetWidth()), 
                                        Vector<3,unsigned int>(phantom.texr->GetWidth(), phantom.texr->GetHeight(), 1));
                
    }
    else {
        PropertyTree tree(yamlSequence);
        PropertyTreeNode* seq = tree.GetRootNode()->GetNodePath("sequence");
        if (!seq) throw Exception(string("No yaml sequence found in file: ") + yamlSequence);
        string name = seq->GetPath("name", string());
        if (name == string("SpinEchoSequence"))
            sequence = new SpinEchoSequence(seq);
        else if (name == string("GradientEchoSequence"))
            sequence = new GradientEchoSequence(seq);
        else if (name == string("EchoPlanarSequence")) 
            sequence = new EchoPlanarSequence(seq);
        else throw Exception(string("Unknown sequence name: ") + name + string(". In file: " + yamlSequence));
    }
}

MRICommandLine::~MRICommandLine() {

}

Phantom MRICommandLine::GetPhantom() {
    return phantom;
}

IMRISequence* MRICommandLine::GetSequence() {
    return sequence;
}

IMRIKernel* MRICommandLine::GetKernel() {
    return kernel;
}
