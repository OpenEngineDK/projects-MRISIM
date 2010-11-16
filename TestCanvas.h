// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_TEST_CANVAS_H_
#define _OE_TEST_CANVAS_H_

#include <Display/ICanvas.h>

using namespace OpenEngine::Display;

/**
 * Short description.
 *
 * @class TestCanvas TestCanvas.h ts/MRISIM/TestCanvas.h
 */
class TestCanvas : public ICanvas {
private:
    bool init;
public:
    TestCanvas(ICanvasBackend* backend) : ICanvas(backend),init(false) {
        
    }
    unsigned int GetWidth() const {
        return backend->GetTexture()->GetWidth();
    };

    unsigned int GetHeight() const {
        return backend->GetTexture()->GetHeight();
    };
    void SetWidth(const unsigned int width) {
        backend->SetDimensions(width, backend->GetTexture()->GetHeight());
    }

    void SetHeight(const unsigned int height) {
        backend->SetDimensions(backend->GetTexture()->GetWidth(), height);
    }
    ITexture2DPtr GetTexture() {
        return backend->GetTexture();
    }
    void Handle(OpenEngine::Display::InitializeEventArg arg) {
        if (init) return;
        backend->Init(100, 100);
        init = true;
    }
    void Handle(OpenEngine::Display::ProcessEventArg arg) {
        backend->Pre();
        
        backend->Post();
    }
    void Handle(OpenEngine::Display::DeinitializeEventArg arg) {
        if (!init) return;
        
        backend->Deinit();
        init = false;
    }
    void Handle(OpenEngine::Display::ResizeEventArg arg) {
        backend->SetDimensions(arg.canvas.GetWidth(), arg.canvas.GetHeight());
    }


};

#endif // _OE_TEST_CANVAS_H_
