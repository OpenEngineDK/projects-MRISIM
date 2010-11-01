// Medical Imaging NetCDF resource.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "MINCResource.h"

#include <Resources/DirectoryManager.h>
#include <Resources/ResourceManager.h>
#include <Resources/File.h>
#include <Logging/Logger.h>
#include <Utils/Convert.h>


namespace OpenEngine {
namespace Resources {

// PLUG-IN METHODS

/**
 * Get the file extension for MINC files.
 */
MINCPlugin::MINCPlugin() {
    this->AddExtension("mnc");
}

/**
 * Create a MINC resource.
 */
MINCResourcePtr MINCPlugin::CreateResource(string file) {
    return MINCResourcePtr(new MINCResource(file));
}


// RESOURCE METHODS

/**
 * Resource constructor.
 */
MINCResource::MINCResource(string file) 
    : file(DirectoryManager::FindFileInPath(file))
  , loaded(false)
  , handle(new mihandle_t())
{
}

/**
 * Resource destructor.
 */
MINCResource::~MINCResource() {
    Unload();
    delete handle;
}

/**
 * 
 *
 * 
 */
void MINCResource::Load() {
    if (loaded) return;

    int result = miopen_volume(file.c_str(), MI2_OPEN_READ, handle); 
    if (result != MI_NOERROR) {
        logger.warning << "Error opening the MINC input file: " << file << logger.end;
        return;
    }
    
    int count;
    miget_volume_voxel_count(*handle, &count);
    logger.info << "voxel count: " << count << logger.end;

    loaded = true;
}

/**
 * Unload the resource.
 * 
 */
void MINCResource::Unload() {
    if (!loaded) return;
    miclose_volume(*handle); 
    loaded = false;
}

/**
 * 
 *
 * 
 */
ITexture3DPtr MINCResource::GetTexture3D() {
    return ITexture3DPtr();
}


} // NS Resources
} // NS OpenEngine
