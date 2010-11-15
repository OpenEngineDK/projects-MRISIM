/// MRI concrete MINC phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "MINCPhantomBuilder.h"
#include "MINCResource.h"

#include <Resources/ResourceManager.h>
#include <Resources/DirectoryManager.h>
#include <Resources/File.h>
#include <Utils/PropertyTree.h>

namespace OpenEngine {
namespace Resources {

using namespace Utils;

MINCPhantomBuilder::MINCPhantomBuilder(string filename): filename(filename) {
}

MINCPhantomBuilder::~MINCPhantomBuilder() {

}
    
Phantom MINCPhantomBuilder::GetPhantom() {
    Phantom phantom;

    filename = DirectoryManager::FindFileInPath(filename);
    string filedir = File::Parent(filename);
    PropertyTree ptree(filename);
    ptree.Reload();

    unsigned int width, height, depth;
    unsigned char* data = NULL;
    if (ptree.HaveNode("voxels")) {
        PropertyTreeNode voxels = ptree.GetNode("voxels");
        phantom.sizeX = voxels.Get("sizeX", 1);
        phantom.sizeY = voxels.Get("sizeY", 1);
        phantom.sizeZ = voxels.Get("sizeZ", 1);
        phantom.offsetX = voxels.Get("offsetX", 0);
        phantom.offsetY = voxels.Get("offsetY", 0);
        phantom.offsetZ = voxels.Get("offsetZ", 0);
        
        width = voxels.Get("width", 1);
        height = voxels.Get("height", 1);
        depth = voxels.Get("depth", 1);
    }

    vector<string> mincs;
    if (ptree.HaveNode("spinPackets")) {
        PropertyTreeNode sp = ptree.GetNode("spinPackets");
        unsigned int count = sp.GetSize();
        phantom.spinPackets = vector<SpinPacket>(count);
        mincs = vector<string>(count);
        for (unsigned int i = 0; i < count; ++i) {
            PropertyTreeNode entry = sp.GetNode(i);
            phantom.spinPackets[i] = SpinPacket(entry.Get("name", string("")),
                                                  entry.Get("t1", 0.0),
                                                  entry.Get("t2", 0.0));
            mincs[i] = entry.Get("minc", string(""));
        }

        unsigned int sz = width*height*depth;
        float* tmp = new float[sz*sizeof(float)];
        data = new unsigned char[sz];
        memset((char*)data, 0, sz);
        for (unsigned int i = 0; i < sz; ++i) {
            tmp[i] = 0.0f;
        }

        for (unsigned int i = 1; i < mincs.size(); ++i) {
            logger.info << "MINCPhantomBuilder: Processing " << mincs[i] << logger.end;
            MINCResourcePtr minc = ResourceManager<MINCResource>::Create(filedir+mincs[i]);
            minc->Load();
            ITexture3DPtr t = minc->GetTexture3D();
            minc->Unload();
            if (t->GetWidth() != width || t->GetHeight() != height || t->GetDepth() != depth) {
                logger.error << "invalid dimensions: " << t->GetWidth() << " x " << t->GetHeight()
                             << " x " << t->GetDepth() << logger.end;
                throw Exception("invalid dimensions");
            }
            else {
                float* mincd = (float*)t->GetVoidDataPtr();
                for (unsigned int j = 0; j < sz; ++j) {
                    if (tmp[j] < mincd[j]) { 
                        tmp[j] = mincd[j];
                        data[j] = i;
                    }
                }
            }
        }
        delete[] tmp;
    }
    phantom.texr = UCharTexture3DPtr(new UCharTexture3D(width, height, depth, 1, data));
    logger.info << "MINCPhantomBuilder: Processing done!" << logger.end;
    return phantom;
}

} // NS Resources
} // NS OpenEngine
