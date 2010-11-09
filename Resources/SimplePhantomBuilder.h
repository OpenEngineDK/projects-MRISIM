// MRI concrete simple phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_SIMPLE_PHANTOM_BUILDER_
#define _MRI_SIMPLE_PHANTOM_BUILDER_

#include "IPhantomBuilder.h"

namespace OpenEngine {
namespace Resources {

class SimplePhantomBuilder: public IPhantomBuilder {
public:
    SimplePhantomBuilder();
    virtual ~SimplePhantomBuilder();
    Phantom GetPhantom();
};

} // NS Resources
} // NS OpenEngine

#endif // _MRI_SIMPLE_PHANTOM_BUILDER_
