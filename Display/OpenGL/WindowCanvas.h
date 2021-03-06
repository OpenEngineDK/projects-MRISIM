// auto windowing on an intensity texture
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _OPENGL_WINDOW_CANVAS_H_
#define _OPENGL_WINDOW_CANVAS_H_

#include <Display/ICanvas.h>
#include <Resources/Texture2D.h>
#include <Core/IListener.h>

namespace OpenEngine {
    namespace Renderers {
        class IRenderer;
    }
    namespace Display {
        class ICanvasBackend;
    }
}

namespace MRI {
  namespace Display {
      namespace OpenGL {

using OpenEngine::Display::ICanvas;
using OpenEngine::Display::ICanvasBackend;
using OpenEngine::Renderers::IRenderer;
using OpenEngine::Resources::UCharTexture2DPtr;
using OpenEngine::Resources::FloatTexture2DPtr;
using OpenEngine::Resources::ITexture2DPtr;
using OpenEngine::Resources::TextureChangedEventArg;
using OpenEngine::Core::IListener;

class WindowCanvas : public ICanvas,
                     public IListener<TextureChangedEventArg> {
private:
    bool init, update;
    IRenderer& renderer;
    FloatTexture2DPtr in;
    UCharTexture2DPtr out;
    unsigned int channels;
    float *min, *max;
public:
    WindowCanvas(ICanvasBackend* backend, FloatTexture2DPtr in,
                 IRenderer& renderer, float scale = 1.0);
    virtual ~WindowCanvas();
  
    void Handle(OpenEngine::Display::InitializeEventArg arg);
    void Handle(OpenEngine::Display::ProcessEventArg arg);
    void Handle(OpenEngine::Display::ResizeEventArg arg);
    void Handle(OpenEngine::Display::DeinitializeEventArg arg);
    void Handle(TextureChangedEventArg arg);
    unsigned int GetWidth() const;
    unsigned int GetHeight() const;
    void SetWidth(const unsigned int width);
    void SetHeight(const unsigned int height);

    OpenEngine::Resources::ITexture2DPtr GetTexture();
};

} // NS OpenGL
} // NS Display
} // NS MRI

#endif // _OPENGL_WINDOW_CANVAS_H_
