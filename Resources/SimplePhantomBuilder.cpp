// MRI concrete simple phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "SimplePhantomBuilder.h"

namespace OpenEngine {
namespace Resources {

SimplePhantomBuilder::SimplePhantomBuilder() {
        
}

SimplePhantomBuilder::~SimplePhantomBuilder() {

}
    
Phantom SimplePhantomBuilder::GetPhantom() {
    Phantom phantom;
    const unsigned int dims = 10; // 10 x 10 x 10 sample
    const unsigned int half = dims/2;
    phantom.sizeX = phantom.sizeY = phantom.sizeZ = 3.0; // 3 mm^3 voxels
    phantom.offsetX = phantom.offsetY = phantom.offsetZ = -half; // origo is in the center of the sample
    vector<SpinPacket> spinPackets(4); // four different spin packet types.
 
    // Tissue Type; T1(ms); T2 (ms); ro; Ki * 10-6
    // Air; 0; 0; 0; 0
    // Connective; 500; 70; 0.77; -9.05
    // CSF; 2569; 329; 1; -9.05
    // Fat; 350; 70; 1; -7 to -8
    spinPackets[0] = SpinPacket("Air", 0.0, 0.0);
    spinPackets[1] = SpinPacket("Connective", 500.0, 70.0);
    spinPackets[2] = SpinPacket("CSF", 2569.0, 329.0);
    spinPackets[3] = SpinPacket("Fat", 350.0, 70.0);

    phantom.spinPackets = spinPackets;

    unsigned char* data = new unsigned char[dims*dims*dims];
    memset((void*)data, 0, dims*dims*dims);
    for (unsigned int i = 0; i < half; ++i) {
        for (unsigned int j = 0; j < half; ++j) {
            for (unsigned int k = 0; k < half; ++k) {
                data[i + j*dims + k*dims*dims] = 1;
            }
        }
    }

    for (unsigned int i = half; i < dims; ++i) {
        for (unsigned int j = half; j < dims; ++j) {
            for (unsigned int k = half; k < dims; ++k) {
                data[i + j*dims + k*dims*dims] = 2;
            }
        }
    }

    for (unsigned int i = 0; i < half; ++i) {
        for (unsigned int j = half; j < dims; ++j) {
            for (unsigned int k = half; k < dims; ++k) {
                data[i + j*dims + k*dims*dims] = 3;
            }
        }
    }

    phantom.texr = UCharTexture3DPtr(new UCharTexture3D(dims, dims, dims, 1, data));
    return phantom;
}

} // NS Resources
} // NS OpenEngine
