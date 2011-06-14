// MRI concrete simple phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_TEST_PHANTOM_BUILDER_
#define _MRI_TEST_PHANTOM_BUILDER_

#include "IPhantomBuilder.h"

namespace MRI {
namespace Resources {

class TestPhantomBuilder: public IPhantomBuilder {
private:
    Vector<3,unsigned int> dims;
    float voxelSize;
public:
    TestPhantomBuilder(Vector<3,unsigned int> dims, float voxelSize = 1.0);
    virtual ~TestPhantomBuilder();
    Phantom GetPhantom();
};

} // NS Resources
} // NS OpenEngine

#endif // _MRI_TEST_PHANTOM_BUILDER_
