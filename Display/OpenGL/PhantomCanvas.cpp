// View the slices of a phantom using color codes.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "PhantomCanvas.h"

#include <Logging/Logger.h>

namespace OpenEngine {
namespace Display {
class ICanvasBackend;
namespace OpenGL {

    PhantomCanvas::PhantomCanvas(ICanvasBackend* backend, Phantom phantom) 
        : ICanvas(backend)
        , init(false)
        , phantom(phantom)
    {
        UCharTexture3DPtr texr = phantom.texr;
        unsigned char* phanData = (unsigned char*)texr->GetVoidDataPtr();

        unsigned char* data = new unsigned char[texr->GetWidth()*texr->GetHeight()*texr->GetDepth()*3];

        vector<Vector<3,unsigned char> > colors(11);
        colors[0] = Vector<3,unsigned char>();
        colors[1] = Vector<3,unsigned char>(255,0,0);
        colors[2] = Vector<3,unsigned char>(0,255,0);
        colors[3] = Vector<3,unsigned char>(0,0,255);
        colors[4] = Vector<3,unsigned char>(255,0,0);
        colors[5] = Vector<3,unsigned char>(0,255,0);
        colors[6] = Vector<3,unsigned char>(0,0,255);
        colors[7] = Vector<3,unsigned char>(255,0,0);
        colors[8] = Vector<3,unsigned char>(0,255,0);
        colors[9] = Vector<3,unsigned char>(0,0,255);
        colors[10] = Vector<3,unsigned char>(0,255,0);
        colors[11] = Vector<3,unsigned char>(0,0,255);

        for (unsigned int i = 0; i < texr->GetWidth()*texr->GetHeight()*texr->GetDepth(); ++i) {
            Vector<3,unsigned char> col = colors[phanData[i]];
            data[i*3] = col[0];
            data[i*3+1] = col[1];
            data[i*3+2] = col[2];
        }

        UCharTexture3DPtr tex(new UCharTexture3D(texr->GetWidth(),
                                                 texr->GetHeight(),
                                                 texr->GetDepth(),
                                                 3,
                                                 data));
        sliceCanvas = new SliceCanvas(backend->Clone(), tex);

        width = sliceCanvas->GetWidth(); 
        height = sliceCanvas->GetHeight();
        logger.info << "width: " << width << logger.end;
        sliceCanvas->SetSlice(50);
        backend->Create(width, height);
    }
    
    PhantomCanvas::~PhantomCanvas() {
        delete sliceCanvas;
    }

    void PhantomCanvas::Handle(Display::InitializeEventArg arg) {
        if (init) return;
        sliceCanvas->Handle(arg);
        backend->Init(width, height);
        init = true;
    }

    void PhantomCanvas::Handle(Display::ProcessEventArg arg) {
        sliceCanvas->Handle(arg);

        backend->Pre();
        
        ITexture2DPtr tex = sliceCanvas->GetTexture();

        Vector<4,int> d(0, 0, width, height);
        glViewport((GLsizei)d[0], (GLsizei)d[1], (GLsizei)d[2], (GLsizei)d[3]);
        OrthogonalViewingVolume volume(-1, 1, 0, width, 0, height);

        // Select The Projection Matrix
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        CHECK_FOR_GL_ERROR();
    
        // Reset The Projection Matrix
        glLoadIdentity();
        CHECK_FOR_GL_ERROR();
    
        // Setup OpenGL with the volumes projection matrix
        Matrix<4,4,float> projMatrix = volume.GetProjectionMatrix();
        float arr[16] = {0};
        projMatrix.ToArray(arr);
        glMultMatrixf(arr);
        CHECK_FOR_GL_ERROR();
    
        // Select the modelview matrix
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        CHECK_FOR_GL_ERROR();
    
        // Reset the modelview matrix
        glLoadIdentity();
        CHECK_FOR_GL_ERROR();
        
        // Get the view matrix and apply it
        Matrix<4,4,float> matrix = volume.GetViewMatrix();
        float f[16] = {0};
        matrix.ToArray(f);
        glMultMatrixf(f);
        CHECK_FOR_GL_ERROR();
        
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);
        GLint texenv;
        glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &texenv);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        const float z = 0.0;
        
        glBindTexture(GL_TEXTURE_2D, tex->GetID());

        CHECK_FOR_GL_ERROR();
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0);
        glVertex3i(0, height, z);
        glTexCoord2f(0.0, 1.0);
        glVertex3i(0, 0, z);
        glTexCoord2f(1.0, 1.0);
        glVertex3i(width, 0, z);
        glTexCoord2f(1.0, 0.0);
        glVertex3i(width, height, z);
        glEnd();
 
        glBindTexture(GL_TEXTURE_2D, 0);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        CHECK_FOR_GL_ERROR();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        CHECK_FOR_GL_ERROR();
        
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, texenv);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_TEXTURE_2D);
            
        backend->Post();
    }
    
    void PhantomCanvas::Handle(Display::ResizeEventArg arg) {
        // backend.SetDimensions(arg.canvas.GetWidth(), arg.canvas.GetHeight());
    }
    
    void PhantomCanvas::Handle(Display::DeinitializeEventArg arg) {
        if (!init) return;
        sliceCanvas->Handle(arg);
        backend->Deinit();
        init = false;
    }

    unsigned int PhantomCanvas::GetWidth() const {
        return width;
    }

    unsigned int PhantomCanvas::GetHeight() const {
        return height;
    }

    void PhantomCanvas::SetWidth(const unsigned int width) {
        // backend.SetDimensions(width, backend.GetHeight());
    }

    void PhantomCanvas::SetHeight(const unsigned int height) {
        // backend.SetDimensions(backend.GetWidth(), height);
    }

    ITexture2DPtr PhantomCanvas::GetTexture() {
        return backend->GetTexture();
    }    
    
    SliceCanvas* PhantomCanvas::GetSliceCanvas() {
        return sliceCanvas;
    }

}
}
}
