// MRI Phantom
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_PHANTOM_
#define _MRI_PHANTOM_

#include <Resources/Texture3D.h>
#include <Utils/PropertyTree.h>
#include <vector>
#include <string>

namespace MRI {
namespace Resources {

using namespace OpenEngine::Resources;
using std::vector;
using std::string;

struct SpinPacket {
    string name;
    float t1, t2, ro;
    SpinPacket(): t1(0.0), t2(0.0), ro(0.0) {}
    SpinPacket(string name, float t1, float t2, float ro): name(name), t1(t1), t2(t2), ro(ro) {}
};

struct Phantom {
public:
    vector<SpinPacket> spinPackets;
    float sizeX, sizeY, sizeZ;  // voxel dimensions in mm
    int offsetX, offsetY, offsetZ;
    UCharTexture3DPtr texr;    

    Phantom();
    Phantom(string filename);
    virtual ~Phantom();

    static void Save(string filename, Phantom phantom);
};

} // NS Resources
} // NS OpenEngine

#endif // _MRI_PHANTOM_
