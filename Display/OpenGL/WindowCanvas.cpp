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
    , update(true)
    , renderer(renderer)
    , in(in)
    , channels(in->GetChannels())
    , min(new float[channels])
    , max(new float[channels])
{
    unsigned char* data = new unsigned char[in->GetWidth()*in->GetHeight()*channels];
    memset(data, 0, in->GetWidth()*in->GetHeight()*sizeof(unsigned char)*channels);
    out = UCharTexture2DPtr(new UCharTexture2D(in->GetWidth(), in->GetHeight(), channels, data));

    backend->Create(in->GetWidth()*scale, in->GetHeight()*scale);
    backend->GetTexture()->SetFiltering(NONE);
    backend->GetTexture()->SetMipmapping(false);
    backend->GetTexture()->SetWrapping(CLAMP);

    out->SetFiltering(NONE);
    out->SetMipmapping(false);
    out->SetWrapping(CLAMP);

    in->ChangedEvent().Attach(*this);

}

WindowCanvas::~WindowCanvas() {
    
}
    
void WindowCanvas::Handle(OpenEngine::Display::InitializeEventArg arg) {
    if (init) return;
    unsigned int width = backend->GetTexture()->GetWidth();
    unsigned int height = backend->GetTexture()->GetHeight();
    logger.info << "w: " << width << " h: " << height << logger.end;
    // if (width == 0 || height == 0) {
    //     width = arg.canvas.GetWidth();
    //     height = arg.canvas.GetHeight();
    // }
    renderer.LoadTexture(in);
    renderer.LoadTexture(out);
    backend->Init(width, height);
    init = true;
}
    
void WindowCanvas::Handle(OpenEngine::Display::ProcessEventArg arg) {
    if (!update) return;
    update = false;
    backend->Pre();
    

    // auto windowing
    for (unsigned int ch = 0; ch < channels; ++ch) {
        min[ch] = INFINITY;
        max[ch] = -INFINITY;
    }

    // fetch max and min values
    float* data = in->GetData();
    for (unsigned int i = 0; i < in->GetWidth()*in->GetHeight(); ++i) {
        for (unsigned int ch = 0; ch < channels; ++ch) {
            min[ch] = fmin(min[ch], data[i*channels+ch]);
            max[ch] = fmax(max[ch], data[i*channels+ch]);
        }
    }         
    
    unsigned char* odata = out->GetData();
    for (unsigned int ch = 0; ch < channels; ++ch) {
        float dist = (max[ch] - min[ch]);
        if (dist != 0.0) { 
            for (unsigned int i = 0; i < out->GetWidth()*out->GetHeight(); ++i) {
                odata[i*channels+ch] = (unsigned char)(255.0f * ((data[i*channels+ch] - min[ch]) / dist));
            } 
        }
    }
    renderer.RebindTexture(out, 0, 0, out->GetWidth(), out->GetHeight());

    // the usual ortho setup
    const unsigned int width = GetWidth();
    const unsigned int height = GetHeight();

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
    GLint texenv;
    glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &texenv);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBindTexture(GL_TEXTURE_2D, out->GetID());
    CHECK_FOR_GL_ERROR();


    glBegin(GL_QUADS);

    glTexCoord2f(0,1);        glVertex2f(0,0);
    glTexCoord2f(1,1);        glVertex2f(width,0);
    glTexCoord2f(1,0);        glVertex2f(width,height);
    glTexCoord2f(0,0);        glVertex2f(0,height);        

    glEnd();
  
    glBindTexture(GL_TEXTURE_2D, 0);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, texenv);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);

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

void WindowCanvas::Handle(TextureChangedEventArg arg) {
    if (arg.resource == in) update = true;
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
