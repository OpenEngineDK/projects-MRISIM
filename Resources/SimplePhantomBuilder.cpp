// MRI concrete simple phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "SimplePhantomBuilder.h"

#include <Logging/Logger.h>


namespace MRI {
namespace Resources {

using namespace std;

SimplePhantomBuilder::SimplePhantomBuilder(unsigned int dims, float voxelSize)
  : dims(dims)
  , voxelSize(voxelSize) {
        
}

SimplePhantomBuilder::~SimplePhantomBuilder() {

}
    
Phantom SimplePhantomBuilder::GetPhantom() {
    Phantom phantom;
    const float half = float(dims)*0.5;
    const float quarter = half*0.5;
    phantom.sizeX = phantom.sizeY = phantom.sizeZ = voxelSize; // m^3 voxels
    phantom.offsetX = phantom.offsetY = phantom.offsetZ = -half; // origo is in the center of the sample
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

    const unsigned int sphereCount = 4;
    pair<Vector<3,float>, float> spheres[sphereCount] = 
        { make_pair(Vector<3,float>(quarter), half*1.25),
          make_pair(Vector<3,float>(quarter*3), half*1.25),

          make_pair(Vector<3,float>(quarter*3, quarter, quarter), quarter),
          make_pair(Vector<3,float>(quarter, quarter*3, quarter), quarter*0.75)
        };

    
    // logger.info << "center: " << c1 << logger.end;
    // logger.info << "diameter: " << d1 << logger.end;

    unsigned char* data = new unsigned char[dims*dims*dims];
    memset((void*)data, 0, dims*dims*dims);
    for (unsigned int i = 0; i < dims; ++i) {
        for (unsigned int j = 0; j < dims; ++j) {
            for (unsigned int k = 0; k < dims; ++k) {
                Vector<3,float> p(i,j,k);
                for (unsigned int l = 0; l < sphereCount; ++l) {
                    Vector<3,float> c = spheres[l].first;
                    float d = spheres[l].second;
                    // logger.info << "length: " << p.GetLength() << logger.end;
                    unsigned int count = 0;
                    if ((p - c).GetLength() < d) {
                        data[i + j*dims + k*dims*dims] = l%4+1;
                        count++;
                    }
                    if (count > 1) // intersections 
                        data[i + j*dims + k*dims*dims] = 4;
                }
            }
        }
    }

    // for (unsigned int i = half; i < dims; ++i) {
    //     for (unsigned int j = half; j < dims; ++j) {
    //         for (unsigned int k = half; k < dims; ++k) {
    //             data[i + j*dims + k*dims*dims] = 2;
    //         }
    //     }
    // }

    // for (unsigned int i = 0; i < half; ++i) {
    //     for (unsigned int j = half; j < dims; ++j) {
    //         for (unsigned int k = half; k < dims; ++k) {
    //             data[i + j*dims + k*dims*dims] = 3;
    //         }
    //     }
    // }

    phantom.texr = UCharTexture3DPtr(new UCharTexture3D(dims, dims, dims, 1, data));
    return phantom;
}

} // NS Resources
} // NS OpenEngine
