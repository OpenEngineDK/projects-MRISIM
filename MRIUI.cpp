#include "MRIUI.h"
#include "MRISIM/ui_MRIUI.h"
// Global

#include <Meta/OpenGL.h>
#include <Meta/Config.h>

#include <Resources/DirectoryManager.h>
#include <Resources/ResourceManager.h>
#include <Resources/SDLFont.h>

// #include <Display/SceneGraphGUI.h>
// #include <Display/SceneNodeGUI.h>
#include <Display/QtEnvironment.h>
#include <Display/RenderCanvas.h>
#include <Display/GridLayout.h>
#include <Display/InspectionWidget.h>

#include <Display/Camera.h>
#include <Display/PerspectiveViewingVolume.h>



#include <Display/OpenGL/TextureCopy.h>

#include <Renderers/OpenGL/Renderer.h>
#include <Renderers/OpenGL/RenderingView.h>

#include <Utils/SimpleSetup.h>
#include <Utils/IInspector.h>

// Local
#include "Resources/MINCResource.h"
#include "Resources/Phantom.h"
#include "Resources/SimplePhantomBuilder.h"
#include "Resources/MINCPhantomBuilder.h"
#include "Display/OpenGL/SliceCanvas.h"
#include "Display/OpenGL/PhantomCanvas.h"

#include "Science/SpinEchoSequence.h"

#include "Science/ImageFFT.h"
#include "Science/CPUFFT.h"

// #include "Science/OpenCLTest.h"

#include "Display/OpenGL/SpinCanvas.h"
#include "Display/OpenGL/WindowCanvas.h"

#undef main // Evil hack :/

using namespace OpenEngine::Logging;
using namespace OpenEngine::Core;
using namespace OpenEngine::Utils;
using namespace OpenEngine::Resources;
using namespace OpenEngine::Scene;
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Renderers::OpenGL;
using namespace OpenEngine::Display;
using namespace OpenEngine::Display::OpenGL;

using namespace MRI::Scene;
using namespace MRI::Science;
using namespace MRI::Display::OpenGL;


namespace OpenEngine {
    namespace Utils {
        namespace Inspection {

ValueList Inspect(SliceCanvas* sc, string name) {
    ValueList values;
    {
        RWValueCall<SliceCanvas, unsigned int > *v
            = new RWValueCall<SliceCanvas, unsigned int >(*sc,
                                                          &SliceCanvas::GetSlice,
                                                          &SliceCanvas::SetSlice);
        v->name = name;
        v->properties[MIN] = 0;
        v->properties[MAX] = sc->GetMaxSlice();
        values.push_back(v);
    }
    return values;
    
}

ValueList Inspect(SpinCanvas* sc) {
    ValueList values;
    {
        RWValueCall<SpinCanvas, unsigned int > *v
            = new RWValueCall<SpinCanvas, unsigned int >(*sc,
                                                          &SpinCanvas::GetSlice,
                                                          &SpinCanvas::SetSlice);
        v->name = "Spin Slice";
        v->properties[MIN] = 0;
        v->properties[MAX] = sc->GetMaxSlice();
        values.push_back(v);
    }
    return values;
}


}}}


void MRIUI::SetupPlugins() {
    DirectoryManager::AppendPath("projects/MRISIM/data/");

    //ResourceManager<IFontResource>::AddPlugin(new CairoFontPlugin());
    ResourceManager<IFontResource>::AddPlugin(new SDLFontPlugin());
    ResourceManager<MINCResource>::AddPlugin(new MINCPlugin());
}

void MRIUI::SetupWall() {

    // wc->AddTextureWithText(phantomCanvas->GetTexture(), "phantom");
    // wc->AddTextureWithText(plot->GetTexture(), "plot");
    // wc->AddTextureWithText(spinCanvas->GetTexture(), "Transverse Spins");
    // wc->AddTextureWithText(fftPlot->GetTexture(), "fft");

    // fourier test stuff

    // ITexture2DPtr t = phantom->CreateSagitalSlice(50);
    // ImageFFT* ifft = new ImageFFT(t, *(new CPUFFT()));
    

    // tl->Load(ifft->GetSrcTexture(), TextureLoader::RELOAD_IMMEDIATE);

    // wc->AddTextureWithText(ifft->GetSrcTexture(), "ImFFT src");

    // WindowCanvas* windowCanvas = new WindowCanvas(new TextureCopy(), ifft->GetStep1Texture(), *r, 1.0f);
    // cq->PushCanvas(windowCanvas);
    // wc->AddTextureWithText(windowCanvas->GetTexture(), "ImFFT step1");

    // windowCanvas = new WindowCanvas(new TextureCopy(), ifft->GetStep2Texture(), *r, 1.0f);
    // cq->PushCanvas(windowCanvas);
    // wc->AddTextureWithText(windowCanvas->GetTexture(), "ImFFT step2");

    // windowCanvas = new WindowCanvas(new TextureCopy(), ifft->GetFFT2DTexture(), *r, 1.0f);
    // cq->PushCanvas(windowCanvas);
    // wc->AddTextureWithText(windowCanvas->GetTexture(), "ImFFT2D");
    // // wc->AddTextureWithText(ifft->GetFFT2DTexture(), "ImFFT2D");

    // windowCanvas = new WindowCanvas(new TextureCopy(), ifft->GetFFT2DInvTexture(), *r, 1.0f);
    // cq->PushCanvas(windowCanvas);
    // wc->AddTextureWithText(windowCanvas->GetTexture(), "ImFFT2DInv");
    // // wc->AddTextureWithText(ifft->GetFFT2DInvTexture(), "ImFFT2DInv");

}

void MRIUI::SetupCanvas() {
    r = new Renderer();
    tl = new TextureLoader(*r);
    r->PreProcessEvent().Attach(*tl);

    wc = new WallCanvas(new TextureCopy(), *r, *tl, font, new GridLayout());

    mouse->MouseMovedEvent().Attach(*wc);
    mouse->MouseButtonEvent().Attach(*wc);

    // push canvases on the queue to get processing time.
    cq = new CanvasQueue();
    cq->PushCanvas(wc);
    frame->SetCanvas(cq);
}

void MRIUI::SetupSim() {
    // --- box phantom ---
    // IPhantomBuilder* pb = new SimplePhantomBuilder(41);
    // Phantom p = pb->GetPhantom();
    
    // ---- brain phantom ---
    IPhantomBuilder* pb = new MINCPhantomBuilder("brain/2/phantom.yaml");
    Phantom p = pb->GetPhantom();
    Phantom::Save("test", p);

    // -- phantom loaded from yaml file ---
    // Phantom p = Phantom("test.yaml");

    // init a canvas that scrolls through the slices of the phantom
    phantomCanvas = new PhantomCanvas(new TextureCopy(), p, 400, 400);
    // tl->Load(phantomCanvas->GetSliceCanvas()->GetSourceTexture());
    cq->PushCanvas(phantomCanvas);
    wc->AddTextureWithText(phantomCanvas->GetTexture(), "phantom");

    // init the simulator and kernel
    kern = new CPUKernel();
    sim = new MRISim(p, kern, NULL);//new SpinEchoSequence(2000.0, 200.0, p));
    engine->InitializeEvent().Attach(*sim);
    engine->ProcessEvent().Attach(*sim);
    engine->DeinitializeEvent().Attach(*sim);
    
    // -- visualise transverse spins, slice by slice --
    spinCanvas = new SpinCanvas(new TextureCopy(), *kern, *r, 400, 400);
    cq->PushCanvas(spinCanvas);
    wc->AddTextureWithText(spinCanvas->GetTexture(), "Transverse Spins");

    // WindowCanvas* windowCanvas = new WindowCanvas(new TextureCopy(), sim->GetKPlane(), *r, 1.0);
    // cq->PushCanvas(windowCanvas);
    // tl->Load(sim->GetKPlane(), TextureLoader::RELOAD_IMMEDIATE);

    // WindowCanvas* windowCanvas2 = new WindowCanvas(new TextureCopy(), sim->GetImagePlane(), *r, 1.0);
    // cq->PushCanvas(windowCanvas2);
    // tl->Load(sim->GetImagePlane(), TextureLoader::RELOAD_IMMEDIATE);

    // plot = new MathGLPlot(400,200);
    // // fftPlot = new MathGLPlot(400,200);

    // tl->Load(plot->GetTexture(), TextureLoader::RELOAD_IMMEDIATE);
    // tl->Load(fftPlot->GetTexture(), TextureLoader::RELOAD_IMMEDIATE);
}

void MRIUI::LoadResources() {
    font = ResourceManager<IFontResource>::Create("Fonts/FreeSansBold.ttf");
    font->Load();
    font->SetSize(24);
    font->SetColor(Vector<3,float>(193.0/256.0,21.0/256.0,21.0/256.0));
}

void MRIUI::Exit() {
    exit(0);
}

void MRIUI::SetupOpenCL() {
    // OpenCLTest test;
    // test.RunKernel();
}

class Flipper {
private:
    CPUKernel* kern;
    SpinCanvas* spinCanvas;
public:
    Flipper(CPUKernel* kern,
            SpinCanvas* spinCanvas): kern(kern), spinCanvas(spinCanvas) {}
    virtual ~Flipper() {}
    void Flip() {
        logger.info << "flip slice: " << spinCanvas->GetSlice() << " 90 deg." << logger.end;
        kern->RFPulse(Math::PI*0.5, spinCanvas->GetSlice());
    }

    void Flop() {
        logger.info << "flip slice: " << spinCanvas->GetSlice() << " 180 deg." << logger.end;
        kern->RFPulse(Math::PI, spinCanvas->GetSlice());
    }

ValueList Inspect() {
    ValueList values;
    {
        ActionValueCall<Flipper> *v =
            new ActionValueCall<Flipper>(*this, &Flipper::Flip);
        v->name = "Flip 90";
        values.push_back(v);
    }

    {
        ActionValueCall<Flipper> *v =
            new ActionValueCall<Flipper>(*this, &Flipper::Flop);
        v->name = "Flip 180";
        values.push_back(v);
    }
    return values;
    
}

};

class Slicer {
private:
    SpinCanvas* spins;
public:
    vector<SliceCanvas*> slices;
    Slicer(SpinCanvas* spins = NULL): spins(spins) {}
    virtual ~Slicer() {}

    void SetSlice(unsigned int slice) {
        for (unsigned int i = 0; i < slices.size(); ++i) {
            slices[i]->SetSlice(slice);
        }
        if (spins)
            spins->SetSlice(slice);
    };

    unsigned int GetSlice() {
        if (slices.empty()) return 0;
        return slices[0]->GetSlice();
    }

    unsigned int GetMaxSlice() {
        if (slices.empty()) return 0;
        return slices[0]->GetMaxSlice();
    }

    ValueList Inspect() {
        ValueList values;
        {
            RWValueCall<Slicer, unsigned int > *v
                = new RWValueCall<Slicer, unsigned int >(*this,
                                                         &Slicer::GetSlice,
                                                         &Slicer::SetSlice);
            v->name = "Slice";
            v->properties[MIN] = 0;
            v->properties[MAX] = GetMaxSlice();
            values.push_back(v);
        }
        return values;
    }
};

MRIUI::MRIUI(QtEnvironment *env) {
    SimpleSetup* setup = new SimpleSetup("MRISIM",env);
    frame = &setup->GetFrame();
    mouse = &setup->GetMouse();
    engine = &setup->GetEngine();
    SetupPlugins();
    LoadResources();


    SetupCanvas();
    SetupWall();
    SetupOpenCL();
    SetupSim();
 
    QApplication *app = env->GetApplication();
    //app->setStyle("plastique");
    //app->setStyle("motif");
    app->setStyle("clearlooks");

    ui = new Ui::MRIUI();
    ui->setupUi(this);
    ui->topLayout->addWidget(env->GetGLWidget());

    Slicer slicer(spinCanvas);
    slicer.slices.push_back(phantomCanvas->GetSliceCanvas());
    ValueList vl = slicer.Inspect();

    ActionValue* av = new ActionValueCall<MRIUI>(*this, &MRIUI::Exit);
    av->name = "Exit";
    vl.push_back(av);
    InspectionWidget *iw = new InspectionWidget("Slice",vl) ;
    iw->setMinimumWidth(200);
    setup->GetEngine().ProcessEvent().Attach(*iw);

    InspectionWidget *iw2 = new InspectionWidget("MRISim", sim->Inspect());
    iw2->setMinimumWidth(300);
    setup->GetEngine().ProcessEvent().Attach(*iw2);


    Flipper flipper(kern, spinCanvas);
    InspectionWidget *iw3 = new InspectionWidget("Flipper", flipper.Inspect());
    iw3->setMinimumWidth(300);
    setup->GetEngine().ProcessEvent().Attach(*iw3);

    QDockWidget *dwI  = new QDockWidget("Slice Inspector",this);
    QDockWidget *dwI2 = new QDockWidget("MRISim Inspector",this);
    QDockWidget *dwI3 = new QDockWidget("Kernel Inspector",this);

    dwI->setWidget(iw);
    dwI2->setWidget(iw2);
    dwI3->setWidget(iw3);
    
    addDockWidget(Qt::RightDockWidgetArea, dwI);
    addDockWidget(Qt::RightDockWidgetArea, dwI2);
    addDockWidget(Qt::RightDockWidgetArea, dwI3);

    ui->menuView->addAction(dwI->toggleViewAction());
    ui->menuView->addAction(dwI2->toggleViewAction());
    ui->menuView->addAction(dwI3->toggleViewAction());
    
    show();
    setup->GetEngine().Start();
}





int main(int argc, char* argv[]) {
    QtEnvironment* env = new QtEnvironment(false, 800, 600, 32, 
                                           FrameOption(), argc, argv);
    MRIUI *ui = new MRIUI(env);
}
