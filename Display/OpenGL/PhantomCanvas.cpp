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
#include <Math/HSLColor.h>
#include <Math/RGBColor.h>

namespace OpenEngine {
namespace Display {
class ICanvasBackend;
namespace OpenGL {

PhantomCanvas::PhantomCanvas(ICanvasBackend* backend, Phantom phantom, unsigned int width, unsigned int height) 
        : ICanvas(backend)
        , init(false)
        , phantom(phantom)
    {
        UCharTexture3DPtr texr = phantom.texr;
        unsigned char* phanData = (unsigned char*)texr->GetVoidDataPtr();
        unsigned char* data = new unsigned char[texr->GetWidth()*texr->GetHeight()*texr->GetDepth()*3];

        vector<Vector<3,unsigned char> > colors(phantom.spinPackets.size());
        HSLColor hsl;
        float phase = 0.0; // hue phase offset in degrees
        float freq  = 45.0 / float(phantom.spinPackets.size()-1); // uniform distribution of hue
        colors[0] = Vector<3,unsigned char>(255);

        for (unsigned int i = 1; i < phantom.spinPackets.size(); ++i) {
            hsl = HSLColor(phase + i * freq, 0.95, 0.6);
            colors[i] = hsl.GetRGB().GetUChar();
        }

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
        tex->SetWrapping(CLAMP);
        tex->SetFiltering(NONE);
        sliceCanvas = new SliceCanvas(backend, tex, width, height);
    }
    
    PhantomCanvas::~PhantomCanvas() {
        delete sliceCanvas;
    }

    void PhantomCanvas::Handle(Display::InitializeEventArg arg) {
        if (init) return;
        sliceCanvas->Handle(arg);
        init = true;
    }

    void PhantomCanvas::Handle(Display::ProcessEventArg arg) {
        sliceCanvas->Handle(arg);
    }
    
    void PhantomCanvas::Handle(Display::ResizeEventArg arg) {
        // backend.SetDimensions(arg.canvas.GetWidth(), arg.canvas.GetHeight());
    }
    
    void PhantomCanvas::Handle(Display::DeinitializeEventArg arg) {
        if (!init) return;
        sliceCanvas->Handle(arg);
        init = false;
    }

    unsigned int PhantomCanvas::GetWidth() const {
        return sliceCanvas->GetWidth();
    }

    unsigned int PhantomCanvas::GetHeight() const {
        return sliceCanvas->GetHeight();
    }

    void PhantomCanvas::SetWidth(const unsigned int width) {
        sliceCanvas->SetWidth(width);
    }

    void PhantomCanvas::SetHeight(const unsigned int height) {
        sliceCanvas->SetHeight(height);
    }

    ITexture2DPtr PhantomCanvas::GetTexture() {
        return sliceCanvas->GetTexture();
    }    
    
    SliceCanvas* PhantomCanvas::GetSliceCanvas() {
        return sliceCanvas;
    }

}
}
}
