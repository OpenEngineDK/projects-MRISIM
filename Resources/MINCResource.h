// Medical Imaging NetCDF resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MINC_RESOURCE_H_
#define _MINC_RESOURCE_H_

#include <minc2.h>

#include <Resources/IResourcePlugin.h>
#include <Resources/IResource.h>
#include <Resources/ITexture3D.h>

#include <string>

namespace OpenEngine {
namespace Resources {

using std::string;

class MINCResource;
typedef boost::shared_ptr<MINCResource> MINCResourcePtr;

/**
 * Medical Imaging NetCDF resource.
 *
 * Loads medical voxel data into OpenEngine Texture resource.
 *
 * @class MINCResource MINCResource.h "MINCResource.h"
 */
class MINCResource : public IResource<MINCResource> {
private:
    // inner material structure
    string file; //!< minc file path
    bool loaded;
    mihandle_t* handle;
public:
    MINCResource(string file);
    virtual ~MINCResource();

    void Load();
    void Unload();
    
    ITexture3DPtr GetTexture3D();
};

/**
 * MINC resource plug-in.
 *
 * @class MINCPlugin MINCResource.h "MINCResource.h"
 */
class MINCPlugin : public IResourcePlugin<MINCResource> {
public:
	MINCPlugin();
    MINCResourcePtr CreateResource(string file);
};


} // NS Resources
} // NS OpenEngine

#endif // _MINC_RESOURCE_H_
