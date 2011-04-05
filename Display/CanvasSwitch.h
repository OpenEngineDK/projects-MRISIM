// Canvas switch
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CANVAS_SWITCH_H_
#define _CANVAS_SWITCH_H_

#include <Display/ICanvas.h>

namespace OpenEngine {
namespace Display {


/**
 * Canvas Switch
 *
 * 
 * @class CanvasSwitch CanvasSwitch.h Display/CanvasSwitch.h
 */
class CanvasSwitch : public ICanvas {
private:
    ICanvas* can;
public:
    CanvasSwitch(ICanvas* canvas);
    virtual ~CanvasSwitch();

    void Handle(Display::InitializeEventArg arg);
    void Handle(Display::ProcessEventArg arg);
    void Handle(Display::ResizeEventArg arg);
    void Handle(Display::DeinitializeEventArg arg);

    unsigned int GetWidth() const;
    unsigned int GetHeight() const;
    void SetWidth(const unsigned int width);
    void SetHeight(const unsigned int height);
    ITexture2DPtr GetTexture();

    void SetCanvas(ICanvas* canvas);
};

} // NS Display
} // NS OpenEngine

#endif // _CANVAS_QUEUE_H_
