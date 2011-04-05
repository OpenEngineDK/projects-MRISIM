// Canvas Initializer
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CANVAS_INITIALIZER_H_
#define _CANVAS_INITIALIZER_H_

#include <Display/CanvasQueue.h>

namespace OpenEngine {
namespace Display {

class InitCanvasQueue : public CanvasQueue {
public:
    InitCanvasQueue() {}
    virtual ~InitCanvasQueue() {}

    void Handle(Display::ProcessEventArg arg) {}
};

} // NS Display
} // NS OpenEngine

#endif // _CANVAS_INITIALIZER_H_
