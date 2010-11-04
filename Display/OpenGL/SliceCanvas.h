// OpenGL canvas to extract axis aligned slices of 3d textures.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _OPENGL_SLICE_CANVAS_H_
#define _OPENGL_SLICE_CANVAS_H_

#include <Display/ICanvas.h>
#include <Resources/ITexture3D.h>
#include <Renderers/OpenGL/Renderer.h>
#include <Display/RenderCanvas.h>

namespace OpenEngine {
namespace Display {
class ICanvasBackend;
namespace OpenGL {

    enum SlicePlane {
        XY,
        YZ,
        XZ
    };

class SliceCanvas : public ICanvas {
private:
    bool init;
    ITexture3DPtr tex;
    SlicePlane plane;
    unsigned int slice, sliceMax;
public:
    SliceCanvas(ICanvasBackend* backend, ITexture3DPtr tex, SlicePlane plane = XY) 
        : ICanvas(backend)
        , init(false)
        , tex(tex)
        , plane(plane)
        , slice(0)
        , sliceMax(0)
    {}

    virtual ~SliceCanvas() {
    }

    void Handle(Display::InitializeEventArg arg) {
        if (init) return;
        
        // hack to bind 3d texture
        Renderers::OpenGL::Renderer r;
        Display::RenderCanvas rc(NULL);
        r.Handle(Renderers::InitializeEventArg(rc));
        r.LoadTexture(tex);

        backend->Init(tex->GetWidth(), tex->GetDepth());//Height());
        sliceMax = tex->GetHeight();
        init = true;
    }

    void Handle(Display::ProcessEventArg arg) {
        backend->Pre();
        
        unsigned int width = GetWidth();
        unsigned int height = GetHeight();

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
        glEnable(GL_TEXTURE_3D);
        GLint texenv;
        glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &texenv);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        const float z = 0.0;

        float texSlice = (float)slice/(float)sliceMax;
        
        glBindTexture(GL_TEXTURE_3D, tex->GetID());
        CHECK_FOR_GL_ERROR();
        glBegin(GL_QUADS);
        glTexCoord3f(0.0, texSlice, 1.0);
        glVertex3f(0.0, 0.0, z);

        glTexCoord3f(0.0, texSlice, 0.0);
        glVertex3f(0, height, z);

        glTexCoord3f(1.0, texSlice, 0.0);
        glVertex3f(width, height, z);

        glTexCoord3f(1.0, texSlice, 1.0);
        glVertex3f(width, 0.0, z);

        glEnd();
 
        glBindTexture(GL_TEXTURE_3D, 0);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        CHECK_FOR_GL_ERROR();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        CHECK_FOR_GL_ERROR();
        
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, texenv);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_TEXTURE_3D);
            
        backend->Post();
    }
    
    void Handle(Display::ResizeEventArg arg) {
        // backend.SetDimensions(arg.canvas.GetWidth(), arg.canvas.GetHeight());
    }
    
    void Handle(Display::DeinitializeEventArg arg) {
        if (!init) return;
        backend->Deinit();
        init = false;
    }

    unsigned int GetWidth() const {
        return tex->GetWidth();
    }

    unsigned int GetHeight() const {
        // return tex->GetHeight();
        return tex->GetDepth();
    }

    void SetWidth(const unsigned int width) {
        // backend.SetDimensions(width, backend.GetHeight());
    }

    void SetHeight(const unsigned int height) {
        // backend.SetDimensions(backend.GetWidth(), height);
    }

    ITexture2DPtr GetTexture() {
        return backend->GetTexture();
    }    

    void SetSlice(unsigned int index) {
        if (index >= sliceMax)
            slice = sliceMax-1;
        else slice = index;
    }

    unsigned int GetSlice() {
        return slice;
    };
};

} // NS OpenGL
} // NS Display
} // NS OpenEngine

#endif // #define _OPENGL_SPLIT_STEREO_CANVAS_H_
