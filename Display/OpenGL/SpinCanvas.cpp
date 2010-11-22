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

namespace MRI {
  namespace Display {
    namespace OpenGL {

using namespace OpenEngine::Display;

SpinCanvas::SpinCanvas(ICanvasBackend* backend, IMRIKernel& kernel, IRenderer& renderer, unsigned int width, unsigned int height)
  : ICanvas(backend)
  , init(false)
  , kernel(kernel)
  , renderer(renderer)
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
  init = true;
}

void SpinCanvas::Handle(OpenEngine::Display::ProcessEventArg arg) {
  backend->Pre();
  
  unsigned int width = GetWidth();
  unsigned int height = GetHeight();

  Vector<4,int> dims(0, 0, width, height);
  glViewport((GLsizei)dims[0], (GLsizei)dims[1], (GLsizei)dims[2], (GLsizei)dims[3]);
  OrthogonalViewingVolume volume(-1, 1, 0, width, 0, height);

  renderer.ApplyViewingVolume(volume);

  Vector<4,float> bgc(0.0, 0.0, 0.0, 1.0);
  glClearColor(bgc[0], bgc[1], bgc[2], bgc[3]);
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  float rad = fmin(width, height) / (10.0*2);
  Phantom phantom = kernel.GetPhantom();
  Vector<3,float>* magnets = kernel.GetMagnets();

  unsigned int w = phantom.texr->GetWidth();
  unsigned int h = phantom.texr->GetHeight();
  unsigned int d = phantom.texr->GetDepth();

  unsigned int slice = d/2;

  for (unsigned int i = 0; i < w; ++i) {
    for (unsigned int j = 0; j < h; ++j) {
      Vector<3,float> p1(2*i*rad + rad, 2*j*rad+rad, 0.0);
      Vector<3,float> p2 = magnets[i + j*w + slice*w*d];
      p2[2] = 0.0;
      if (p2*p2 > 0.0)
	p2.Normalize();
      p2 = p2*rad + p1;
      Line l(p1, p2);
      renderer.DrawLine(l, Vector<3,float>(1.0));
    }
  }

  // Select The Projection Matrix
  // glMatrixMode(GL_PROJECTION);
  // glPushMatrix();
  // CHECK_FOR_GL_ERROR();
    
  // Reset The Projection Matrix
  // glLoadIdentity();
  // CHECK_FOR_GL_ERROR();
    
  // Setup OpenGL with the volumes projection matrix
  // Matrix<4,4,float> projMatrix = volume.GetProjectionMatrix();
  // float arr[16] = {0};
  // projMatrix.ToArray(arr);
  // glMultMatrixf(arr);
  // CHECK_FOR_GL_ERROR();
    
  // Select the modelview matrix
  // glMatrixMode(GL_MODELVIEW);
  // glPushMatrix();
  // CHECK_FOR_GL_ERROR();
    
  // Reset the modelview matrix
  // glLoadIdentity();
  // CHECK_FOR_GL_ERROR();
        
  // Get the view matrix and apply it
  // Matrix<4,4,float> matrix = volume.GetViewMatrix();
  // float f[16] = {0};
  // matrix.ToArray(f);
  // glMultMatrixf(f);
  // CHECK_FOR_GL_ERROR();
        
  // glDisable(GL_DEPTH_TEST);
  // GLint texenv;
  // glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &texenv);
  // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  // const float z = 0.0;

  // float texSlice = 2.0; //(float)slice/(float)sliceMax;
        
  // glBindTexture(GL_TEXTURE_3D, tex->GetID());
  // CHECK_FOR_GL_ERROR();
  // glBegin(GL_QUADS);
  // glTexCoord3f(0.0, texSlice, 1.0);
  // glVertex3f(0.0, 0.0, z);

  // glTexCoord3f(0.0, texSlice, 0.0);
  // glVertex3f(0, height, z);

  // glTexCoord3f(1.0, texSlice, 0.0);
  // glVertex3f(width, height, z);

  // glTexCoord3f(1.0, texSlice, 1.0);
  // glVertex3f(width, 0.0, z);

  // glEnd();
 
  // glBindTexture(GL_TEXTURE_3D, 0);

  // glMatrixMode(GL_PROJECTION);
  // glPopMatrix();
  // CHECK_FOR_GL_ERROR();
  // glMatrixMode(GL_MODELVIEW);
  // glPopMatrix();
  // CHECK_FOR_GL_ERROR();
        
  // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, texenv);
  // glEnable(GL_DEPTH_TEST);
  // glDisable(GL_TEXTURE_3D);
            
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

} // NS OpenGL
} // NS Display
} // NS MRI
