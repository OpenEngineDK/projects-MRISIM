// Visualise a slice of spins.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _OPENGL_SPIN_CANVAS_CANVAS_H_
#define _OPENGL_SPIN_CANVAS_CANVAS_H_

#include <Display/ICanvas.h>
#include <Resources/ITexture2D.h>

namespace OpenEngine {
  namespace Renderers {
    class IRenderer;
  }
  namespace Display {
    class ICanvasBackend;
  }
}

namespace MRI {
  namespace Science {
    class IMRIKernel;
  }
  namespace Display {
    namespace OpenGL {

using OpenEngine::Display::ICanvas;
using OpenEngine::Display::ICanvasBackend;
using OpenEngine::Renderers::IRenderer;
using Science::IMRIKernel;

class SpinCanvas : public ICanvas {
private:
  bool init;
  IMRIKernel& kernel;
  IRenderer& renderer;
public:
  SpinCanvas(ICanvasBackend* backend, IMRIKernel& kernel, 
	     IRenderer& renderer, unsigned int width = 0, unsigned int height = 0);
  virtual ~SpinCanvas();
  
  void Handle(OpenEngine::Display::InitializeEventArg arg);
  void Handle(OpenEngine::Display::ProcessEventArg arg);
  void Handle(OpenEngine::Display::ResizeEventArg arg);
  void Handle(OpenEngine::Display::DeinitializeEventArg arg);
  unsigned int GetWidth() const;
  unsigned int GetHeight() const;
  void SetWidth(const unsigned int width);
  void SetHeight(const unsigned int height);

  OpenEngine::Resources::ITexture2DPtr GetTexture();
};

} // NS OpenGL
} // NS Display
} // NS MRI

#endif // _OPENGL_SPIN_CANVAS_CANVAS_H_
