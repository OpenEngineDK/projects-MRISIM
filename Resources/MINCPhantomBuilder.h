// MRI concrete MINC phantom builder
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_MINC_PHANTOM_BUILDER_
#define _MRI_MINC_PHANTOM_BUILDER_

#include "IPhantomBuilder.h"

#include <string>

namespace MRI {
namespace Resources {

using std::string;

class MINCPhantomBuilder: public IPhantomBuilder {
private:
    string filename;
public:
    MINCPhantomBuilder(string filename);
    virtual ~MINCPhantomBuilder();
    Phantom GetPhantom();
};

} // NS Resources
} // NS OpenEngine

#endif // _MRI_MINC_PHANTOM_BUILDER_
