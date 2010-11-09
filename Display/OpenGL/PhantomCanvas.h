// View the slices of a phantom using color codes.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _OPENGL_PHANTOM_CANVAS_H_
#define _OPENGL_PHANTOM_CANVAS_H_

#include "../../Resources/Phantom.h"
#include "SliceCanvas.h"

namespace OpenEngine {
namespace Display {
class ICanvasBackend;
namespace OpenGL {

class PhantomCanvas : public ICanvas {
private:
    bool init;
    Phantom phantom;
    SliceCanvas* sliceCanvas;
    unsigned int width, height;
public:
    PhantomCanvas(ICanvasBackend* backend, Phantom phantom);
    virtual ~PhantomCanvas();
    void Handle(Display::InitializeEventArg arg);    
    void Handle(Display::ProcessEventArg arg);
    void Handle(Display::ResizeEventArg arg);
    void Handle(Display::DeinitializeEventArg arg);
    unsigned int GetWidth() const;
    unsigned int GetHeight() const;
    void SetWidth(const unsigned int width);
    void SetHeight(const unsigned int height);
    ITexture2DPtr GetTexture();
    SliceCanvas* GetSliceCanvas();
};

} // NS OpenGL
} // NS Display
} // NS OpenEngine

#endif // #define _OPENGL_PHANTOM_CANVAS_H_
