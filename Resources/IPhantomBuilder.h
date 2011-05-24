// MRI Interface for constructing phantoms 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_INTERFACE_PHANTOM_BUILDER_
#define _MRI_INTERFACE_PHANTOM_BUILDER_

#include "Phantom.h"

namespace MRI {
namespace Resources {

class IPhantomBuilder {
public:
    virtual ~IPhantomBuilder() {};
    virtual Phantom GetPhantom() = 0;
};

} // NS Resources
} // NS OpenEngine

#endif // _MRI_INTERFACE_PHANTOM_BUILDER_
