// OE stuff
#include <Resources/DirectoryManager.h>

//MRI stuff
#include "Resources/Phantom.h"
#include "Resources/Sample3DTexture.h"

#include "Science/MRISim.h"
#include "Resources/Sample3DTexture.h"

#include <Logging/Logger.h>
#include <Logging/StreamLogger.h>

#include "MRICommandLine.h"

#include <cstdio>
#include <fstream>

using namespace MRI::Science;
using namespace MRI::Resources;

using namespace OpenEngine::Logging;

int main(int argc, char* argv[]) {
    DirectoryManager::AppendPath("projects/MRISIM/Science/");
    DirectoryManager::AppendPath("projects/MRISIM/data/");

    Logger::AddLogger(new StreamLogger(new fstream("output.log", fstream::out | fstream::trunc)));

    printf("Initializing ...\n");

    MRICommandLine cmdl(argc, argv);
    IMRISequence* seq = cmdl.GetSequence();
    Phantom p = cmdl.GetPhantom();
    IMRIKernel* kern = cmdl.GetKernel();

    MRISim* sim = new MRISim(p, kern, seq);
    sim->Start();
    unsigned int dstep = seq->GetNumPoints() / 100;
    unsigned int sum = 0;
    if (dstep == 0) dstep = 1;

    printf("Simulating using %s kernel...\n", kern->GetName().c_str());

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
