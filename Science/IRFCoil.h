// Interface for an RF coil.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_INTERFACE_RF_COIL_
#define _MRI_INTERFACE_RF_COIL_

#include <Math/Vector.h>

namespace MRI {
namespace Science {

using OpenEngine::Math::Vector;

/**
 * MRI Interface for an RF coil.
 *
 * @class IRFCoil IRFCoil.h MRISIM/Science/IRFCoil.h
 */
class IRFCoil {
public:
    virtual ~IRFCoil() {}
    virtual Vector<3,float> GetSignal(const float time) = 0;
};

}
}

#endif //_MRI_INTERFACE_RF_COIL_
