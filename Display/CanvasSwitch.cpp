// Canvas Queue
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "CanvasSwitch.h"

#include <Logging/Logger.h>

namespace OpenEngine {
namespace Display {

    CanvasSwitch::CanvasSwitch(ICanvas* can):
        can(can) {

    }
    
    CanvasSwitch::~CanvasSwitch() {

    }

    void CanvasSwitch::Handle(Display::InitializeEventArg arg) {
        ((IListener<Display::InitializeEventArg>*)can)->Handle(arg);
    }
    
    void CanvasSwitch::Handle(Display::ProcessEventArg arg) {
        // logger.info << "process: " << can << logger.end;

        ((IListener<Display::ProcessEventArg>*)can)->Handle(arg);
    }
    
    void CanvasSwitch::Handle(Display::ResizeEventArg arg) {
        ((IListener<Display::ResizeEventArg>*)can)->Handle(arg);
    }
    
    void CanvasSwitch::Handle(Display::DeinitializeEventArg arg) {
        ((IListener<Display::DeinitializeEventArg>*)can)->Handle(arg);
    }

    unsigned int CanvasSwitch::GetWidth() const {
        return can->GetWidth();
    }
    
    unsigned int CanvasSwitch::GetHeight() const {
        return can->GetHeight();
    }
    
    void CanvasSwitch::SetWidth(const unsigned int width) {
        can->SetWidth(width);
    }
    
    void CanvasSwitch::SetHeight(const unsigned int height) {
        can->SetHeight(height);
    }
    
    ITexture2DPtr CanvasSwitch::GetTexture() {
        return can->GetTexture();
    }

    void CanvasSwitch::SetCanvas(ICanvas* c) {
        can = c;
    }

} // NS Display
} // NS OpenEngine
