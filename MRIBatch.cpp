// OE stuff
#include <Resources/DirectoryManager.h>

//MRI stuff
#include "Resources/Phantom.h"
#include "Resources/SimplePhantomBuilder.h"
#include "Resources/Sample3DTexture.h"

#include "Science/SpinEchoSequence.h"
#include "Science/ExcitationPulseSequence.h"
#include "Science/TestRFCoil.h"

#include "Science/MRISim.h"
#include "Science/CPUKernel.h"
#include "Science/OpenCLKernel.h"

#include "Resources/Sample3DTexture.h"

#include <cstdio>

using namespace MRI::Scene;
using namespace MRI::Science;
using namespace MRI::Resources;

int main(int argc, char* argv[]) {
    DirectoryManager::AppendPath("projects/MRISIM/Science/");
    DirectoryManager::AppendPath("projects/MRISIM/data/");

    bool useCPU = false;
    string yamlSequence;

    unsigned int phantomSize = 20;

    for (int i=1;i<argc;i++) {
        if (strcmp(argv[i],"-cpu") == 0)
            useCPU = true;
        else {
            unsigned int f = strtol(argv[i], NULL, 10);
            if (f > 0)
                phantomSize = f;
        }
    }

    printf("Initializing ...\n");
    // load kernel
    IMRIKernel* kern;
    if (useCPU)
        kern = new CPUKernel();
    else
        kern = new OpenCLKernel();


    // load phantom
    IPhantomBuilder* pb = new SimplePhantomBuilder(phantomSize);
    Phantom p = pb->GetPhantom();

    // load sequence
    IMRISequence* seq = NULL;
    vector<pair<double, MRIEvent> > l;
    if (yamlSequence.empty()) {
        seq = new SpinEchoSequence(2500.0f, 50.0f, p.sizeX * 1e-3 * float(p.texr->GetWidth()));
    }
    else {
        // seq = new ListSequence(l);
        // seq->LoadFromYamlFile(yamlSequence);
    }

    MRISim* sim = new MRISim(p, kern, seq);
    sim->Start();
    unsigned int dstep = seq->GetNumPoints() / 100;
    unsigned int sum = 0;
    if (dstep == 0) dstep = 1;

    printf("Simulating ...\n");

    while (sim->IsRunning()) {
        sim->Simulate(dstep);
        //printf(".");
        sum += 1;
        printf("\r\033[2K");
        printf("percentage: %d", sum);
        fflush(stdout);
    }
    printf("\nReconstructing Samples ...");
    fflush(stdout);
    printf(" done\n");


    vector<complex<float> > samples = seq->GetSampler().GetSamples();    
    vector<complex<float> > image = seq->GetSampler().GetReconstructedSamples();    

    Vector<3,unsigned int> dims = seq->GetSampler().GetDimensions();
    printf("Saving samples and reconstructed image w: %d h: %d d: %d ...", dims[0], dims[1], dims[2]);
    fflush(stdout);

    Sample3DTexture* stex = new Sample3DTexture(samples, dims, false);
    stex->Save("samples");

    Sample3DTexture* itex = new Sample3DTexture(image, seq->GetSampler().GetDimensions(), false);
    itex->Save("image"); 

    printf(" done\n");
}
