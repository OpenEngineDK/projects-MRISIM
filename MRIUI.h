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
#include <Core/IEngine.h>

#include "Science/MRISim.h"
#include "Science/CPUKernel.h"
#include "Science/CartesianFFT.h"
#include "Science/SpinEchoSequence.h"

#include "Display/CanvasSwitch.h"


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


namespace MRI { 
    namespace Scene { class SpinNode; }
    namespace Display {
        namespace OpenGL { class SpinCanvas; }
    }
}


using namespace OpenEngine::Resources;
using namespace OpenEngine::Display;
using namespace OpenEngine::Devices;
using namespace OpenEngine::Display::OpenGL;
using namespace MRI::Display::OpenGL;
using namespace MRI::Science;
using namespace OpenEngine::Science;
using namespace OpenEngine::Core;


class MRIUI : public QMainWindow {
    Q_OBJECT;
    Ui::MRIUI* ui;
    
    IFontResourcePtr font;
    IFrame* frame;
    IEngine* engine;
    IMouse* mouse;

    MRISim* sim;
    CPUKernel* kern;
    SpinEchoSequence* seq;
    MINCResourcePtr phantom;
    SliceCanvas *samplesCanvas, *fftCanvas;
    PhantomCanvas *phantomCanvas;
    SpinCanvas* spinCanvas;
    CartesianFFT* fft;
    WallCanvas* wcSim, *wcRF; 
    CanvasSwitch* cSwitch;
    CanvasQueue* cq;
    MathGLPlot* plot;
    MathGLPlot* fftPlot;
    IRenderer* r;
    TextureLoader* tl;
    MRI::Scene::SpinNode *spinNode;

    QDockWidget *dwI, *dwI1, *dwI2, *dwI3, *dwI4, *dwI5;
    
    void SetupPlugins();
    void SetupCanvas();
    void SetupWall();
    void SetupSim();
    void SetupRF();
    void SetupOpenCL();
    void LoadResources();

    void Exit();

public slots:
    void SetSimView(bool toggle);
    void SetRFView(bool toggle);
public:
    MRIUI(OpenEngine::Display::QtEnvironment *env);    

};




#endif // _OE_M_R_I_U_I_H_
