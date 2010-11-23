// Visualise a slice of spins.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "SpinCanvas.h"

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

SpinCanvas::SpinCanvas(ICanvasBackend* backend, IMRIKernel& kernel, IRenderer& renderer, 
                       unsigned int width, unsigned int height)
    : ICanvas(backend)
    , init(false)
    , kernel(kernel)
    , renderer(renderer)
    , slice(0)
    , sliceMax(kernel.GetPhantom().texr->GetDepth())

{
    backend->Create(width, height);
}
    
SpinCanvas::~SpinCanvas() {
    
}
    
void SpinCanvas::Handle(OpenEngine::Display::InitializeEventArg arg) {
    if (init) return;
    unsigned int width = backend->GetTexture()->GetWidth();
    unsigned int height = backend->GetTexture()->GetHeight();
    if (width == 0 || height == 0) {
        width = arg.canvas.GetWidth();
        height = arg.canvas.GetHeight();
    }
    backend->Init(width, height);

    tex = ResourceManager<ITexture2D>::Create("magnet.png");
    tex->Load();
    tex->SetMipmapping(true);
    tex->SetFiltering(BILINEAR);
    renderer.LoadTexture(tex);
    init = true;
}
    
void SpinCanvas::Handle(OpenEngine::Display::ProcessEventArg arg) {
    backend->Pre();
    
    const unsigned int width = GetWidth();
    const unsigned int height = GetHeight();
    
    // the usual ortho setup
    Vector<4,int> dims(0, 0, width, height);
    glViewport((GLsizei)dims[0], (GLsizei)dims[1], (GLsizei)dims[2], (GLsizei)dims[3]);
    OrthogonalViewingVolume volume(-1, 1, 0, width, 0, height);
    renderer.ApplyViewingVolume(volume);

    Vector<4,float> bgc(.8);
    glClearColor(bgc[0], bgc[1], bgc[2], bgc[3]);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    const Phantom phantom = kernel.GetPhantom();
    const Vector<3,float>* magnets = kernel.GetMagnets();
    const unsigned char *data = phantom.texr->GetData();
    const unsigned int w = phantom.texr->GetWidth();
    const unsigned int h = phantom.texr->GetHeight();
    const unsigned int d = phantom.texr->GetDepth();
    const float rad = fmin(width/(w*2), height/(h*2));

    // the initial quad to be rotated and translated
    const Vector<2,float> _p1(-rad, -rad);
    const Vector<2,float> _p2 = _p1 + Vector<2,float>(2*rad,0.0);
    const Vector<2,float> _p3 = _p2 + Vector<2,float>(0.0,2*rad);
    const Vector<2,float> _p4 = _p1 + Vector<2,float>(0.0,2*rad);
  

    // setup GL
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    // GLint texenv;
    // glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &texenv);
    // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBindTexture(GL_TEXTURE_2D, tex->GetID());
    CHECK_FOR_GL_ERROR();

    for (unsigned int i = 0; i < w; ++i) {
        for (unsigned int j = 0; j < h; ++j) {

            if (data[i + j*w + slice*w*d] == 0)
                continue;

            // Vector<3,float> p1(2*i*rad + rad, 2*j*rad+rad, 0.0);
            // Vector<3,float> p2 = magnets[i + j*w + slice*w*d];
            // p2[2] = 0.0;
            // if (p2*p2 > 0.0)
            //     p2.Normalize();
            // p2 = p2*rad + p1;
            // Line l(p1, p2);
            // renderer.DrawLine(l, Vector<3,float>());

            // calculate the rotation
            // since we normalize using three components we get a
            // scaling based on the equilibrium magnetization
            Vector<3,float> m = magnets[i + j*w + slice*w*d];
            m.Normalize(); 
            Matrix<2,2,float> rot(m[0], -m[1],
                                  m[1], m[0]);

            // rotate
            Vector<2,float> p1 = rot*_p1;
            Vector<2,float> p2 = rot*_p2;
            Vector<2,float> p3 = rot*_p3;
            Vector<2,float> p4 = rot*_p4;

            // translate
            p1 += Vector<2,float>(rad + 2*i*rad, rad + 2*j*rad);
            p2 += Vector<2,float>(rad + 2*i*rad, rad + 2*j*rad);
            p3 += Vector<2,float>(rad + 2*i*rad, rad + 2*j*rad);
            p4 += Vector<2,float>(rad + 2*i*rad, rad + 2*j*rad);

            // draw
            glBegin(GL_QUADS);
            glColor3f(1.0, 0.0, 0.0);
            glTexCoord2f(0.0, 0.0);
            glVertex3f(p1[0], p1[1], 0.0);
            glTexCoord2f(1.0, 0.0);
            glVertex3f(p2[0], p2[1], 0.0);
            glTexCoord2f(1.0, 1.0);
            glVertex3f(p3[0], p3[1], 0.0);
            glTexCoord2f(0.0, 1.0);
            glVertex3f(p4[0], p4[1], 0.0);
            glEnd();
        }
    }
  
    glBindTexture(GL_TEXTURE_2D, 0);
    // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, texenv);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
            
    backend->Post();
}
    
void SpinCanvas::Handle(OpenEngine::Display::ResizeEventArg arg) {
    // backend->SetDimensions(arg.canvas.GetWidth(), arg.canvas.GetHeight());
}
    
void SpinCanvas::Handle(OpenEngine::Display::DeinitializeEventArg arg) {
    if (!init) return;
    backend->Deinit();
    init = false;
}

unsigned int SpinCanvas::GetWidth() const {
    return backend->GetTexture()->GetWidth();
}

unsigned int SpinCanvas::GetHeight() const {
    return backend->GetTexture()->GetHeight();
}

void SpinCanvas::SetWidth(const unsigned int width) {
    backend->SetDimensions(width, backend->GetTexture()->GetHeight());
}

void SpinCanvas::SetHeight(const unsigned int height) {
    backend->SetDimensions(backend->GetTexture()->GetWidth(), height);
}

ITexture2DPtr SpinCanvas::GetTexture() {
    return backend->GetTexture();
}    

void SpinCanvas::SetSlice(unsigned int index) {
    if (index >= sliceMax)
        slice = sliceMax-1;
    else slice = index;
}
    
unsigned int SpinCanvas::GetSlice() {
    return slice;
}

unsigned int SpinCanvas::GetMaxSlice() {
    return sliceMax;
}

} // NS OpenGL
} // NS Display
} // NS MRI
