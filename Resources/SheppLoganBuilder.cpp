// MRI concrete shepp logan phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "SheppLoganBuilder.h"

#include <Logging/Logger.h>

#include <fstream>

namespace MRI {
namespace Resources {

using namespace std;

SheppLoganBuilder::SheppLoganBuilder(string rawfile, unsigned int dims, float voxelSize)
    : rawfile(rawfile)
    , dims(dims)
    , voxelSize(voxelSize) {
        
}

SheppLoganBuilder::~SheppLoganBuilder() {

}
    
Phantom SheppLoganBuilder::GetPhantom() {
    Phantom phantom;
    const unsigned int voxels = dims * dims * dims;
    const float half = float(dims)*0.5;
    phantom.sizeX = phantom.sizeY = phantom.sizeZ = voxelSize; // mm^3 voxels
    phantom.offsetX = phantom.offsetY = phantom.offsetZ = -half; // origo is in the center of the sample
    vector<SpinPacket> spinPackets(4); // four different spin packet types.
 
    // Tissue Type; T1(ms); T2 (ms); ro; Ki * 10-6
    // Air; 0; 0; 0; 0
    // Connective; 500; 70; 0.77; -9.05
    // CSF; 2569; 329; 1; -9.05
    // Fat; 350; 70; 1; -7 to -8
    spinPackets[0] = SpinPacket("Air", 0.0, 0.0, 0.0);
    spinPackets[1] = SpinPacket("CSF", 2.569, 0.329, 1.0);
    spinPackets[2] = SpinPacket("Fat", 0.350, 0.070, 1.0);
    spinPackets[3] = SpinPacket("Gray matter", 0.833, 0.083, 0.86);

    phantom.spinPackets = spinPackets;

    unsigned char* data = new unsigned char[voxels];
    unsigned short* in_buffer = new unsigned short[voxels];

    ifstream fin(rawfile.c_str());
    fin.read((char*)in_buffer, voxels * sizeof(short));
    fin.close();

    for (unsigned int i = 0; i < voxels; ++i) {
        unsigned short in = in_buffer[i];
        switch (in) {
        case 0:
            data[i] = 0;
            break;
        case 65535:
            data[i] = 1;
            break;
        case 13106:
        case 13107:
            data[i] = 2;
            break;
        case 19660:
            data[i] = 3;
            break;
        default:
            logger.warning << "unknown value in shepp logan raw source: " << in << logger.end;
            data[i] = 0;
        }
    }
    phantom.texr = UCharTexture3DPtr(new UCharTexture3D(dims, dims, dims, 1, data));
    return phantom;
}

} // NS Resources
} // NS OpenEngine
