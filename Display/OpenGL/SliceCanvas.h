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
#include <Display/OrthogonalViewingVolume.h>

#include <Logging/Logger.h>

namespace OpenEngine {
namespace Display {
class ICanvasBackend;
namespace OpenGL {

    enum SlicePlane {
        XY,
        YZ,
        XZ
    };

using Renderers::OpenGL::Renderer;

class SliceCanvas : public ICanvas, public IListener<Texture3DChangedEventArg> {
private:
    bool init, updateSlice;
    ITexture3DPtr tex;
    SlicePlane plane;
    unsigned int slice, sliceMax;
    unsigned int width, height;
    GLuint texid;
public:
    SliceCanvas(ICanvasBackend* backend, ITexture3DPtr tex, unsigned int w = 0, unsigned int h = 0, SlicePlane plane = XY) 
        : ICanvas(backend)
        , init(false)
        , updateSlice(true)
        , tex(tex)
        , plane(plane)
        , slice(0)
        , sliceMax(tex->GetHeight())
        , width(tex->GetWidth())
        , height(tex->GetDepth())
        , texid(0)
    {
        tex->ChangedEvent().Attach(*this);
        if (w != 0 && h != 0) {
            width = w;
            height = h;
        }
        backend->Create(width, height);
    }

    virtual ~SliceCanvas() {
    }

    void Handle(Texture3DChangedEventArg arg) {
        updateSlice = true;
    }

    unsigned int GetMaxSlice() { return sliceMax; }
    
    void Handle(Display::InitializeEventArg arg) {
        if (init) return;
        
        // hack to bind 3d texture
        // Renderers::OpenGL::Renderer r;
        // Display::RenderCanvas rc(NULL);
        // r.Handle(Renderers::InitializeEventArg(rc));
        // r.LoadTexture(tex);

        backend->Init(width, height);//Height());
        init = true;
    }

    void Handle(Display::ProcessEventArg arg) {
        if (!updateSlice) return;
        updateSlice = false;
        backend->Pre();

        if (texid) {
            glDeleteTextures(1, &texid);
            texid = 0;
        }
        unsigned int twidth = tex->GetWidth();
        unsigned int theight = tex->GetHeight();
        void* data = ((char*)tex->GetVoidDataPtr()) + (slice * twidth * theight * tex->GetChannels() * tex->GetChannelSize());

        
        GLint internalFormat = Renderer::GLInternalColorFormat(tex->GetColorFormat());
        GLenum colorFormat = Renderer::GLColorFormat(tex->GetColorFormat());

        glGenTextures(1, &texid);
        CHECK_FOR_GL_ERROR();
        // logger.info << "gentex: " << texid << logger.end;
        // logger.info << "width: " << twidth << " height: " << theight << logger.end;

        glBindTexture(GL_TEXTURE_2D, texid);
        CHECK_FOR_GL_ERROR();

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        GLint filter;
        if (tex->GetFiltering() == NONE) 
            filter = GL_NEAREST;
        else 
            filter = GL_LINEAR;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, tex->GetWrapping());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, tex->GetWrapping());
        CHECK_FOR_GL_ERROR();

        glTexImage2D(GL_TEXTURE_2D,
                     0, // mipmap level
                     internalFormat,
                     twidth,
                     theight,
                     0, // border
                     colorFormat,
                     tex->GetType(),
                     data);
        glBindTexture(GL_TEXTURE_2D, 0);
        CHECK_FOR_GL_ERROR();
        
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
        glDisable(GL_COLOR_MATERIAL);
        glEnable(GL_TEXTURE_2D);
        // glEnable(GL_TEXTURE_3D);
        GLint texenv;
        glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &texenv);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        const float z = 0.0;

        float texSlice = float(slice)/float(sliceMax);
        
        // glBindTexture(GL_TEXTURE_3D, tex->GetID());
        glBindTexture(GL_TEXTURE_2D, texid);
        CHECK_FOR_GL_ERROR();

        glBegin(GL_QUADS);
        glColor3f(0.0,0.0,0.0);
        // glTexCoord3f(0.0, texSlice, 1.0);
        glTexCoord2f(0.0, 1.0);
        glVertex3f(0.0, 0.0, z);

        // glTexCoord3f(0.0, texSlice, 0.0);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(0, height, z);

        //glTexCoord3f(1.0, texSlice, 0.0);
        glTexCoord2f(1.0, 0.0);
        glVertex3f(width, height, z);

        //glTexCoord3f(1.0, texSlice, 1.0);
        glTexCoord2f(1.0, 1.0);
        glVertex3f(width, 0.0, z);

        glEnd();
 
        //glBindTexture(GL_TEXTURE_3D, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        CHECK_FOR_GL_ERROR();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        CHECK_FOR_GL_ERROR();
        
        glEnable(GL_DEPTH_TEST);
        //glDisable(GL_TEXTURE_3D);
        glDisable(GL_TEXTURE_2D);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, texenv);
            
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
        return width;
    }

    unsigned int GetHeight() const {
        return height;
    }

    void SetWidth(const unsigned int width) {
        this->width = width;
        backend->SetDimensions(width, GetHeight());
    }

    void SetHeight(const unsigned int height) {
        this->height = height;
        backend->SetDimensions(GetWidth(), height);
    }

    ITexture2DPtr GetTexture() {
        return backend->GetTexture();
    }    

    void SetSlice(unsigned int index) {
        if (index >= sliceMax)
            slice = sliceMax-1;
        else slice = index;
        updateSlice = true;
    }

    unsigned int GetSlice() {
        return slice;
    };

    ITexture3DPtr GetSourceTexture() {
        return tex;
    }

};

} // NS OpenGL
} // NS Display
} // NS OpenEngine

#endif // #define _OPENGL_SPLIT_STEREO_CANVAS_H_
