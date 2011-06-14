// MRI concrete simple phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "TestPhantomBuilder.h"

#include <Logging/Logger.h>


namespace MRI {
namespace Resources {

using namespace std;

TestPhantomBuilder::TestPhantomBuilder(Vector<3,unsigned int> dims, float voxelSize)
  : dims(dims)
  , voxelSize(voxelSize) {
        
}

TestPhantomBuilder::~TestPhantomBuilder() {

}
    
Phantom TestPhantomBuilder::GetPhantom() {
    Phantom phantom;
    phantom.sizeX = phantom.sizeY = phantom.sizeZ = voxelSize; // m^3 voxels
    phantom.offsetX = - 0.5 * dims[0];
    phantom.offsetY = - 0.5 * dims[1];
    phantom.offsetZ = - 0.5 * dims[2]; // origo is in the center of the sample
    vector<SpinPacket> spinPackets(5); // four different spin packet types.
 
    // Tissue Type; T1(ms); T2 (ms); ro; Ki * 10-6
    // Air; 0; 0; 0; 0
    // Connective; 500; 70; 0.77; -9.05
    // CSF; 2569; 329; 1; -9.05
    // Fat; 350; 70; 1; -7 to -8
    spinPackets[0] = SpinPacket("Air", 0.0, 0.0, 0.0, 0.0);
    spinPackets[1] = SpinPacket("White matter", 0.500, 0.070, 0.061, 0.77);
    spinPackets[2] = SpinPacket("CSF", 2.569, 0.329, 0.058, 1.0);
    spinPackets[3] = SpinPacket("Fat", 0.350, 0.070, 0.058, 1.0);
    spinPackets[4] = SpinPacket("Grey matter", 0.833, 0.083, 0.069, 0.86);

    phantom.spinPackets = spinPackets;

    unsigned char* data = new unsigned char[dims[0]*dims[1]*dims[2]];
    for (unsigned int i = 0; i < dims[0]; ++i) {
        for (unsigned int j = 0; j < dims[1]; ++j) {
            for (unsigned int k = 0; k < dims[2]; ++k) {
                if (i > dims[0] / 2)
                    data[i + j*dims[0] + k*dims[0]*dims[1]] = 1;
                else
                    data[i + j*dims[0] + k*dims[0]*dims[1]] = 2;
            }
        }
    }
    
    phantom.texr = UCharTexture3DPtr(new UCharTexture3D(dims[0], dims[1], dims[2], 1, data));
    return phantom;
}

} // NS Resources
} // NS OpenEngine
