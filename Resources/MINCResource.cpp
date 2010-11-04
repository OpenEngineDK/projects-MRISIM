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
{
}

/**
 * Resource destructor.
 */
MINCResource::~MINCResource() {
    Unload();
}

/**
 * 
 *
 * 
 */
void MINCResource::Load() {
    if (loaded) return;

    int result = miopen_volume(file.c_str(), MI2_OPEN_READ, &handle); 
    if (result != MI_NOERROR) {
        logger.warning << "Error opening the MINC input file: " << file << logger.end;
        return;
    }

    // char** names = new char*[3];
    // names[0] = "x_space";
    // names[1] = "y_space";
    // names[2] = "z_space";
    
    // miset_apparent_dimension_order_by_name(handle, 3, names);



    
    int count;
    miget_volume_voxel_count(handle, &count);
    logger.info << "voxel count: " << count << logger.end;

    // atomic type of the resulting texture (float, int)
    miclass_t klass;
    miget_data_class(handle, &klass);
    logger.info << "data class: " << klass << logger.end;

    // data type of a voxel (float, uchar, ...)
    // convert this type into class type by scaling.
    mitype_t type;
    miget_data_type(handle, &type);
    logger.info << "voxel type: " << type << logger.end;

    misize_t vsize;
    miget_data_type_size(handle, &vsize);
    logger.info << "voxel size in bytes: " << vsize << logger.end;

    midimhandle_t dims[3];
    result = miget_volume_dimensions(handle, MI_DIMCLASS_SPATIAL, 
                                     MI_DIMATTR_ALL, MI_DIMORDER_FILE, 
                                     3, dims);

    if (result != 3) {
        logger.warning << "Only three spatial dimensions supported. Volume has " << result << logger.end;
        return;
    }

    w = h = d = 0;
    // hack: in our files dimensions are given in z,y,x order, so we reverse it.
    miget_dimension_size(dims[0], &w);
    miget_dimension_size(dims[1], &h);
    miget_dimension_size(dims[2], &d);

    // midimhandle_t* dims_app = new midimhandle_t[3];
    // dims_app[0] = dims[2];
    // dims_app[1] = dims[1];
    // dims_app[2] = dims[0];

    // miset_apparent_dimension_order (handle, 3, dims_app);
    logger.info << "dimensions: " << w << " x " << h << " x " << d << logger.end;

    
    // count records
    int recs = 0;
    miget_record_length(handle, &recs);
    logger.info << "record length: " << recs << logger.end;

    // count labels
    int lbls = 0;
    miget_number_of_defined_labels(handle, &lbls);
    logger.info << "# of labels: " << lbls << logger.end;

    // count attributes
    milisthandle_t lhandle;
    milist_start(handle, "", 0, &lhandle);
    char* path = new char[255];
    char* name = new char[255];
    while (milist_attr_next(handle, lhandle, path, 255, name, 255) == MI_NOERROR) {
        logger.info << "path: " << string(path) << " name: " << string(name) << logger.end;
    }
    milist_finish(lhandle);
    delete path;
    delete name;

    // char* nm;
    // miget_dimension_name(dims[2], &nm);
    // string hest(nm);
    // mifree_name(nm);
    // logger.info << "dimension name: " << hest << logger.end;

    char* space_name;
    miget_space_name(handle, &space_name);
    logger.info << "space name: " << string(space_name) << logger.end;
    mifree_name(space_name);

    loaded = true;
}

/**
 * Unload the resource.
 * 
 */
void MINCResource::Unload() {
    if (!loaded) return;
    miclose_volume(handle); 
    loaded = false;
}

/**
 * 
 *
 * 
 */
ITexture3DPtr MINCResource::GetTexture3D() {
    if (!loaded) return ITexture3DPtr();
    if (tex) return tex;
    
    const unsigned int size = w*h*d*sizeof(float);

    float* data = new float[size];;
    tex = FloatTexture3DPtr(new FloatTexture3D(w,h,d,1,data));

    string rawfile = file + string(".raw");
    
    if (File::Exists(rawfile)) {
        logger.info << "MINC: Reading raw float data." << logger.end;
        ifstream* f = File::Open(rawfile);
        f->read((char*)data, size);
        f->close();
        delete f;
    }
    else {
        for (unsigned int x = 0; x < w; ++x) {
            for (unsigned int y = 0; y < h; ++y) {
                for (unsigned int z = 0; z < d; ++z) {
                    const unsigned long loc[3] = {x,y,z}; 
                    double val;
                    miget_real_value(handle, loc, 3, &val);
                    data[x + y*w + z*w*h] = val;
                }
            }
        }
        ofstream out(rawfile.c_str());
        out.write((char*)data, size);
        out.close();
    }
    return tex;
}

ITexture2DPtr MINCResource::CreateTransverseSlice(unsigned int index) {
    if (!loaded) return ITexture2DPtr();
    
    // clamp x-index
    if (index >= w) 
        index = w-1;

    float* data = new float[h*d*sizeof(float)];;
    FloatTexture2DPtr slice = FloatTexture2DPtr(new FloatTexture2D(h,d,1,data));

    unsigned int x = index;
    for (unsigned int y = 0; y < h; ++y) {
        for (unsigned int z = 0; z < d; ++z) {
                const unsigned long loc[3] = {x,y,z}; 
                double val;
                miget_real_value(handle, loc, 3, &val);
                data[y+z*h] = val;
        }
    }
    return slice;
}

ITexture2DPtr MINCResource::CreateSagitalSlice(unsigned int index) {
    if (!loaded) return ITexture2DPtr();
    // clamp z-index
    if (index >= d) 
        index = d-1;

    float* data = new float[w*h*sizeof(float)];;
    FloatTexture2DPtr slice = FloatTexture2DPtr(new FloatTexture2D(h,w,1,data));

    unsigned int z = index;
    for (unsigned int x = 0; x < w; ++x) {
        for (unsigned int y = 0; y < h; ++y) {
                const unsigned long loc[3] = {x,y,z}; 
                double val;
                miget_real_value(handle, loc, 3, &val);
                data[y+x*h] = val;
        }
    }
    return slice;
}

ITexture2DPtr MINCResource::CreateCoronalSlice(unsigned int index) {
    if (!loaded) return ITexture2DPtr();
    
    // clamp y-index
    if (index >= h) 
        index = h-1;

    float* data = new float[w*d*sizeof(float)];;
    FloatTexture2DPtr slice = FloatTexture2DPtr(new FloatTexture2D(d,w,1,data));

    unsigned int y = index;
    for (unsigned int x = 0; x < w; ++x) {
        for (unsigned int z = 0; z < d; ++z) {
                const unsigned long loc[3] = {x,y,z}; 
                double val;
                miget_real_value(handle, loc, 3, &val);
                data[z+x*d] = val;
        }
    }
    return slice;
}

} // NS Resources
} // NS OpenEngine
