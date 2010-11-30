// auto windowing on an intensity texture
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "WindowCanvas.h"

#include "../../Science/MRISim.h"
#include "../../Resources/Phantom.h"

#include <Meta/OpenGL.h>
#include <Display/OrthogonalViewingVolume.h>
#include <Renderers/IRenderer.h>
#include <Resources/ResourceManager.h>

namespace MRI {
namespace Display {
namespace OpenGL {

using namespace OpenEngine::Display;
using namespace OpenEngine::Resources;

WindowCanvas::WindowCanvas(ICanvasBackend* backend, FloatTexture2DPtr in, IRenderer& renderer, 
                           float scale)
    : ICanvas(backend)
    , init(false)
    , renderer(renderer)
    , in(in)
{
    backend->Create(in->GetWidth()*scale, in->GetHeight()*scale);
    backend->GetTexture()->SetFiltering(NONE);
    backend->GetTexture()->SetMipmapping(false);
    backend->GetTexture()->SetWrapping(CLAMP);

    in->SetFiltering(NONE);
    in->SetMipmapping(false);
    in->SetWrapping(CLAMP);
}
    
WindowCanvas::~WindowCanvas() {
    
}
    
void WindowCanvas::Handle(OpenEngine::Display::InitializeEventArg arg) {
    if (init) return;
    unsigned int width = backend->GetTexture()->GetWidth();
    unsigned int height = backend->GetTexture()->GetHeight();
    if (width == 0 || height == 0) {
        width = arg.canvas.GetWidth();
        height = arg.canvas.GetHeight();
    }
    renderer.LoadTexture(in);
    backend->Init(width, height);
    init = true;
}
    
void WindowCanvas::Handle(OpenEngine::Display::ProcessEventArg arg) {
    backend->Pre();
    
    const unsigned int width = GetWidth();
    const unsigned int height = GetHeight();
    
    // the usual ortho setup
    Vector<4,int> dims(0, 0, width, height);
    glViewport((GLsizei)dims[0], (GLsizei)dims[1], (GLsizei)dims[2], (GLsizei)dims[3]);
    OrthogonalViewingVolume volume(-1, 1, 0, width, 0, height);
    renderer.ApplyViewingVolume(volume);

    Vector<4,float> bgc(1.0);
    glClearColor(bgc[0], bgc[1], bgc[2], bgc[3]);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // setup GL
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    // GLint texenv;
    // glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &texenv);
    // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBindTexture(GL_TEXTURE_2D, in->GetID());
    CHECK_FOR_GL_ERROR();
    
    float min = INFINITY;
    float max = -INFINITY;

    float* data = in->GetData();
    for (unsigned int i = 0; i < in->GetWidth(); ++i) {
        for (unsigned int j = 0; j < in->GetHeight(); ++j) {
            min = fmin(min, data[i+j*in->GetHeight()]);
            max = fmax(max, data[i+j*in->GetHeight()]);
        }
    } 
    float scale = 1.0/(max-min);
    //scale = 1.0;
    // draw
    // glEnable(GL_BLEND);
    // glBlendFunc(GL_SRC_ALPHA, GL_CONSTANT_COLOR);
    // glBlendColor(fabs(min), fabs(min), fabs(min), 1.0);
    glBegin(GL_QUADS);
    glColor3f(scale, scale, scale);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(width, 0.0, 0.0);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(width, height, 0.0);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(0.0, height, 0.0);
    glEnd();
  
    glBindTexture(GL_TEXTURE_2D, 0);
    // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, texenv);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    // glDisable(GL_BLEND);
            
    backend->Post();
}
    
void WindowCanvas::Handle(OpenEngine::Display::ResizeEventArg arg) {
    // backend->SetDimensions(arg.canvas.GetWidth(), arg.canvas.GetHeight());
}
    
void WindowCanvas::Handle(OpenEngine::Display::DeinitializeEventArg arg) {
    if (!init) return;
    backend->Deinit();
    init = false;
}

unsigned int WindowCanvas::GetWidth() const {
    return backend->GetTexture()->GetWidth();
}

unsigned int WindowCanvas::GetHeight() const {
    return backend->GetTexture()->GetHeight();
}

void WindowCanvas::SetWidth(const unsigned int width) {
    backend->SetDimensions(width, backend->GetTexture()->GetHeight());
}

void WindowCanvas::SetHeight(const unsigned int height) {
    backend->SetDimensions(backend->GetTexture()->GetWidth(), height);
}

ITexture2DPtr WindowCanvas::GetTexture() {
    return backend->GetTexture();
}    

} // NS OpenGL
} // NS Display
} // NS MRI
