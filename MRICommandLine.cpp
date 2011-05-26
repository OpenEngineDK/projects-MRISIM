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
#include "Science/OpenCLKernel.h"
#include "Science/SpinEchoSequence.h"
#include "Science/ExcitationPulseSequence.h"
#include "Science/TestRFCoil.h"
#include "Resources/SimplePhantomBuilder.h"

#include <Utils/PropertyTree.h>
#include <Utils/PropertyTreeNode.h>

using namespace MRI::Resources;
using namespace MRI::Science;

MRICommandLine::MRICommandLine(int argc, char* argv[])
    : sequence(NULL)
    , kernel(NULL)
{
    bool useCPU = false;
    string yamlSequence, yamlPhantom;
    unsigned int phantomSize = 20;
    
    for (int i=1;i<argc;i++) {
        if (strcmp(argv[i],"-cpu") == 0)
            useCPU = true;
        else if (strcmp(argv[i],"-phantom") == 0) {
            if (i + 1 < argc)
                yamlPhantom = string(argv[i+1]);
        }
        else if (strcmp(argv[i],"-sequence") == 0) {
            if (i + 1 < argc)
                yamlSequence = string(argv[i+1]);
        }
        else {
            unsigned int f = strtol(argv[i], NULL, 10);
            if (f > 0)
                phantomSize = f;
        }
    }

    // load kernel
    kernel;
    if (useCPU)
        kernel = new CPUKernel();
    else
        kernel = new OpenCLKernel();

    // load phantom
    if (yamlPhantom.empty()) {
        IPhantomBuilder* pb = new SimplePhantomBuilder(phantomSize);
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
