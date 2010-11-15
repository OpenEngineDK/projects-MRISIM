// MRI Phantom
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "Phantom.h"

#include <Resources/DirectoryManager.h>
#include <Resources/File.h>

#include <fstream>

namespace OpenEngine {
namespace Resources {

using namespace Utils;
using namespace std;

Phantom::Phantom() {

}

Phantom::Phantom(string filename) {
    filename = DirectoryManager::FindFileInPath(filename);
    string filedir = File::Parent(filename);
    PropertyTree ptree(filename);
    ptree.Reload();
    if (ptree.HaveNode("voxels")) {
        PropertyTreeNode voxels = ptree.GetNode("voxels");
        sizeX = voxels.Get("sizeX", 1);
        sizeY = voxels.Get("sizeY", 1);
        sizeZ = voxels.Get("sizeZ", 1);
        offsetX = voxels.Get("offsetX", 0);
        offsetY = voxels.Get("offsetY", 0);
        offsetZ = voxels.Get("offsetZ", 0);
        
        unsigned int width = voxels.Get("width", 1);
        unsigned int height = voxels.Get("height", 1);
        unsigned int depth = voxels.Get("depth", 1);
        
        unsigned int bytesPerVoxel = voxels.Get("bytesPerVoxel", 1);
        unsigned int sz = width*height*depth;
        string rawfile = voxels.Get("rawfile", string(""));
        unsigned char* data = new unsigned char[sz];
        ifstream in(rawfile.c_str());

        if (bytesPerVoxel == 1) {
             in.read((char*)data, sz);
        }
        else if (bytesPerVoxel == 2) {
            unsigned short buffer;
            for (unsigned int i = 0; i < sz; ++i) {
                in.read((char*)&buffer, bytesPerVoxel);
                if (buffer > 255) throw Exception("Voxel value does not fit in unsigned char.");
                data[i] = buffer;
            }
        }
        else if (bytesPerVoxel == 4) {
            unsigned int buffer;
            for (unsigned int i = 0; i < sz; ++i) {
                in.read((char*)&buffer, bytesPerVoxel);
                if (buffer > 255) throw Exception("Voxel value does not fit in unsigned char.");
                data[i] = buffer;
            }
        }
        else {
            throw Exception("bytesPerVoxel must be 1, 2, or 4.");
        }

        texr = UCharTexture3DPtr(new UCharTexture3D(width, height, depth, 1, data));
    }

    if (ptree.HaveNode("spinPackets")) {
        PropertyTreeNode sp = ptree.GetNode("spinPackets");
        unsigned int count = sp.GetSize();
        spinPackets = vector<SpinPacket>(count);
        for (unsigned int i = 0; i < count; ++i) {
            PropertyTreeNode entry = sp.GetNode(i);
            spinPackets[i] = SpinPacket(entry.Get("name", string("")),
                                        entry.Get("t1", 0.0),
                                        entry.Get("t2", 0.0));
        }
    }
}

Phantom::~Phantom() {

}

void Phantom::Save(string filename, Phantom phantom) {
    string indent("");
    string yamlfile = filename + ".yaml";
    string rawfile = filename + ".raw";
    ofstream out(yamlfile.c_str());

    out << "voxels:\n";
    out << "  sizeX: " << phantom.sizeX << "\n";
    out << "  sizeY: " << phantom.sizeY << "\n";
    out << "  sizeZ: " << phantom.sizeZ << "\n";
    out << "  offsetX: " << phantom.offsetX << "\n";
    out << "  offsetY: " << phantom.offsetY << "\n";
    out << "  offsetZ: " << phantom.offsetZ << "\n";
    out << "  width: " << phantom.texr->GetWidth() << "\n";
    out << "  height: " << phantom.texr->GetHeight() << "\n";
    out << "  depth: " << phantom.texr->GetDepth() << "\n";    
    out << "  rawfile: " << rawfile << "\n";
    out << "\n";

    out << "spinPackets:\n";
    for (unsigned int i = 0; i < phantom.spinPackets.size(); ++i) {
        SpinPacket sp = phantom.spinPackets[i];
        out << "  - name: " << sp.name << "\n";
        out << "    t1: " << sp.t1 << "\n";
        out << "    t2: " << sp.t2 << "\n";
    }
    out.close();

    ofstream out2(rawfile.c_str());
    out2.write((char*)phantom.texr->GetVoidDataPtr(), 
              phantom.texr->GetWidth()*phantom.texr->GetHeight()*phantom.texr->GetDepth());
    out2.close();
}

} // NS Resources
} // NS OpenEngine

