// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_M_R_I_U_I_H_
#define _OE_M_R_I_U_I_H_

#include <QtGui>

#include <Resources/IFontResource.h>
#include <Display/IFrame.h>
#include "Resources/MINCResource.h"
#include <Devices/IMouse.h>
#include <Science/MathGLPlot.h>
#include <Display/WallCanvas.h>
#include <Display/CanvasQueue.h>

namespace Ui { class MRIUI; }

namespace OpenEngine { 
namespace Display {
namespace OpenGL {
class SliceCanvas;
class PhantomCanvas;
}
class QtEnvironment; 

}
}


namespace MRI { namespace Scene { class SpinNode; }}

using namespace OpenEngine::Resources;
using namespace OpenEngine::Display;
using namespace OpenEngine::Devices;
using namespace OpenEngine::Display::OpenGL;
using namespace OpenEngine::Science;

class MRIUI : public QMainWindow {
    Q_OBJECT;
    Ui::MRIUI* ui;
    
    IFontResourcePtr font;
    IFrame* frame;
    MINCResourcePtr phantom;
    IMouse* mouse;
    SliceCanvas *sliceCanvas;
    PhantomCanvas *phantomCanvas;
    WallCanvas* wc;
    CanvasQueue* cq;
    MathGLPlot* plot;
    MathGLPlot* fftPlot;
    IRenderer* r;
    TextureLoader* tl;
    MRI::Scene::SpinNode *spinNode;

    
    void SetupPlugins();
    void SetupCanvas();
    void LoadResources();

    void Exit();
public:
    MRIUI(OpenEngine::Display::QtEnvironment *env);    

};




#endif // _OE_M_R_I_U_I_H_
