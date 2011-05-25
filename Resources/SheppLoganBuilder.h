// MRI concrete shepp logan phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_SHEPP_LOGAN_PHANTOM_BUILDER_
#define _MRI_SHEPP_LOGAN_PHANTOM_BUILDER_

#include "IPhantomBuilder.h"

#include <string>

namespace MRI {
namespace Resources {

using std::string;

class SheppLoganBuilder: public IPhantomBuilder {
private:
    string rawfile;
    unsigned int dims;
    float voxelSize;
public:
    SheppLoganBuilder(string rawsource, unsigned int dims, float voxelSize);
    virtual ~SheppLoganBuilder();
    Phantom GetPhantom();
};

} // NS Resources
} // NS OpenEngine

#endif // _MRI_SIMPLE_PHANTOM_BUILDER_
