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

#include <Utils/IInspector.h>

// Local
#include "Resources/MINCResource.h"
#include "Resources/Phantom.h"
#include "Resources/SimplePhantomBuilder.h"
#include "Resources/MINCPhantomBuilder.h"
#include "Resources/SheppLoganBuilder.h"
#include "Resources/Sample3DTexture.h"

#include "Science/SamplerVisualizer.h"

#include "Display/OpenGL/SliceCanvas.h"
#include "Display/OpenGL/PhantomCanvas.h"
#include "Display/InitCanvasQueue.h"



#include "Scene/SpinNode.h"

#include "Science/SpinEchoSequence.h"
#include "Science/ExcitationPulseSequence.h"

#include "Science/ImageFFT.h"
#include "Science/CPUFFT.h"

#include "Science/RFTester.h"
#include "Science/TestRFCoil.h"

// #include "Science/OpenCLTest.h"

#include "Display/OpenGL/SpinCanvas.h"
#include "Display/OpenGL/WindowCanvas.h"

#include "MRICommandLine.h"

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
using namespace MRI::Resources;
using namespace MRI::Display::OpenGL;



ExcitationPulseSequence *rfTestSequence = NULL;

void MRIUI::SetupPlugins() {
    ResourceManager<IFontResource>::AddPlugin(new SDLFontPlugin());
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

    wcSim = new WallCanvas(new TextureCopy(), *r, *tl, font, new GridLayout());
    wcRF = new WallCanvas(new TextureCopy(), *r, *tl, font, new GridLayout());

    // push canvases on the queue to get processing time.

    cq = new CanvasQueue();
    frame->SetCanvas(cq);

    cSwitch = new CanvasSwitch(wcSim);
    cq->PushCanvas(cSwitch);
 
    InitCanvasQueue* icq = new InitCanvasQueue();
    icq->PushCanvas(wcSim);
    icq->PushCanvas(wcRF);
    cq->PushCanvas(icq);

}

void MRIUI::SetupSim() {
    const unsigned int texWidth = 300;
    const unsigned int texHeight = 300;

    // init a canvas that scrolls through the slices of the phantom
    phantomCanvas = new PhantomCanvas(new TextureCopy(), phantom, texWidth, texHeight);
    // tl->Load(phantomCanvas->GetSliceCanvas()->GetSourceTexture());
    cq->PushCanvas(phantomCanvas);
    wcSim->AddTextureWithText(phantomCanvas->GetTexture(), "phantom");

    // --- init the simulator and kernel ---    
    DirectoryManager::AppendPath("projects/MRISIM/Science/");
    
    sim = new MRISim(phantom, kern, seq);

    engine->InitializeEvent().Attach(*sim);
    engine->ProcessEvent().Attach(*sim);
    engine->DeinitializeEvent().Attach(*sim);
    
    // --- visualise transverse spins, slice by slice ---
    spinCanvas = new SpinCanvas(new TextureCopy(), *kern, *r, texWidth, texHeight);
    cq->PushCanvas(spinCanvas);
    wcSim->AddTextureWithText(spinCanvas->GetTexture(), "Transverse Spins");

    // --- visualise the output samples ---

    SamplerVisualizer* sviz = new SamplerVisualizer(seq->GetSampler());
    seq->GetSampler().SamplesChangedEvent().Attach(*sviz);

    // Sample3DTexture* sampleTex = new Sample3DTexture(seq->GetSampler().GetSamples(), seq->GetSampler().GetDimensions(), true);
    // seq->GetSampler().SamplesChangedEvent().Attach(*sampleTex);
    // samplesCanvas = new SliceCanvas(new TextureCopy(), Sample3DTexturePtr(sampleTex), texWidth, texHeight);
    samplesCanvas = new SliceCanvas(new TextureCopy(), sviz->GetSamplesTexture(), texWidth, texHeight);
    cq->PushCanvas(samplesCanvas);
    wcSim->AddTextureWithText(samplesCanvas->GetTexture(), "Samples");

    // --- reconstruct and visualize ---
    // fft = new CartesianFFT(*(new CPUFFT()), sim->GetSamples(), sim->GetSampleDimensions(), true);
    // Sample3DTexture* imageTex = new Sample3DTexture(seq->GetSampler().GetReconstructedSamples(), seq->GetSampler().GetDimensions(), true);
    // fftCanvas = new SliceCanvas(new TextureCopy(), Sample3DTexturePtr(imageTex), texWidth, texHeight);
    fftCanvas = new SliceCanvas(new TextureCopy(), sviz->GetImageTexture(), texWidth, texHeight);
    cq->PushCanvas(fftCanvas);
    wcSim->AddTextureWithText(fftCanvas->GetTexture(), "Reconstruction");
        
    
    // RenderCanvas *rc = new RenderCanvas(new TextureCopy(), Vector<2,int>(200,200));
    // Renderer *rend = new Renderer();

    // Camera *cam = new Camera(*(new PerspectiveViewingVolume()));
    // cam->SetPosition(Vector<3,float>(10,10,10));
    // cam->LookAt(Vector<3,float>(0,0,0));
    // rc->SetViewingVolume(cam);
    
    // SpinNode *sn = new SpinNode(); 
    // sn->Mp = kern->GetMagnets();

    // RenderNode *sn = kern->GetRenderNode();

    // rc->SetScene(sn);
    // rc->SetRenderer(rend);
    // RenderingView *rv = new RenderingView();
    // rend->ProcessEvent().Attach(*rv);
    // rend->InitializeEvent().Attach(*rv);
    // cq->PushCanvas(rc);

    // wc->AddTextureWithText(rc->GetTexture(), "Spin Node");
    // wcSim->AddTextureWithText(rc->GetTexture(), "RF Node");


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

void MRIUI::SetupRF() {
    TestRFCoil* rfcoil = new TestRFCoil(.005, 1.0, GYRO_RAD, 1000 * Math::PI);
    RFTester* tester = new RFTester(*(new CPUFFT()), rfcoil, 600, 200);
    tl->Load(tester->GetTimeTexture(), TextureLoader::RELOAD_IMMEDIATE);
    tl->Load(tester->GetFrequencyTexture(), TextureLoader::RELOAD_IMMEDIATE);
    wcRF->AddTextureWithText(tester->GetTimeTexture(), "RF Pulse Time Domain");
    wcRF->AddTextureWithText(tester->GetFrequencyTexture(), "RF Pulse Frequency Domain");
    tester->RunTest();

    InspectionWidget *iw4 = new InspectionWidget("RFTester", tester->Inspect());
    InspectionWidget *iw5 = new InspectionWidget("RFCoil", rfcoil->Inspect());
    iw4->setMinimumWidth(300);
    iw5->setMinimumWidth(300);
    engine->ProcessEvent().Attach(*iw4);
    engine->ProcessEvent().Attach(*iw5);

    dwI4 = new QDockWidget("RFTester", this);
    dwI4->setWidget(iw4);
    
    dwI5 = new QDockWidget("RFCoil", this);
    dwI5->setWidget(iw5);

    rfTestSequence = new ExcitationPulseSequence(rfcoil);
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
    IMRIKernel* kern;
    SpinCanvas* spinCanvas;
public:
    Flipper(IMRIKernel* kern,
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


    void SetGradientX(float g) {
        Vector<3,float> gv = kern->GetGradient();
        gv[0] = g;
        kern->SetGradient(gv);
    }

    void SetGradientY(float g) {
        Vector<3,float> gv = kern->GetGradient();
        gv[1] = g;
        kern->SetGradient(gv);
    }

     void SetGradientZ(float g) {
        Vector<3,float> gv = kern->GetGradient();
        gv[2] = g;
        kern->SetGradient(gv);
    }

    float GetGradientX() {
        return kern->GetGradient()[0];
    }

    float GetGradientY() {
        return kern->GetGradient()[1];
    }

    float GetGradientZ() {
        return kern->GetGradient()[2];
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

    const float gradientMin = -0.3;
    const float gradientMax =  0.3;
    const float gradientStep =  0.01;

    {
        RWValueCall<Flipper, float> *v
            = new RWValueCall<Flipper, float>(*this,
                                             &Flipper::GetGradientX,
                                             &Flipper::SetGradientX);
        v->name = "Gradient x";
        v->properties[MIN] = gradientMin;
        v->properties[MAX] = gradientMax;
        v->properties[STEP] = gradientStep;
        values.push_back(v);
    }

    {
        RWValueCall<Flipper, float> *v
            = new RWValueCall<Flipper, float>(*this,
                                             &Flipper::GetGradientY,
                                             &Flipper::SetGradientY);
        v->name = "Gradient y";
        v->properties[MIN] = gradientMin;
        v->properties[MAX] = gradientMax;
        v->properties[STEP] = gradientStep;
        values.push_back(v);
    }

    {
        RWValueCall<Flipper, float> *v
            = new RWValueCall<Flipper, float>(*this,
                                             &Flipper::GetGradientZ,
                                             &Flipper::SetGradientZ);
        v->name = "Gradient z";
        v->properties[MIN] = gradientMin;
        v->properties[MAX] = gradientMax;
        v->properties[STEP] = gradientStep;
        values.push_back(v);
    }


    return values;
    
}

};

class Slicer {
private:
    SpinCanvas* spins;
    CartesianFFT* fft;
public:
    vector<SliceCanvas*> slices;
    Slicer(SpinCanvas* spins = NULL, CartesianFFT* fft = NULL): spins(spins), fft(fft) {}
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

    
    void Recon() {
        // seq->GetSampler().Reconstruct();
        //if (fft) fft->ReconstructSlice(GetSlice());
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
        {
            ActionValueCall<Slicer> *v =
                new ActionValueCall<Slicer>(*this, &Slicer::Recon);
            v->name = "Reconstruct Slice";
            values.push_back(v);
        }
        return values;
    }
};

MRIUI::MRIUI(QtEnvironment *env, IMRISequence* sequence, IMRIKernel* kernel, Phantom phantom, SimpleSetup* setup) {
    // ---- brain phantom ---
    // IPhantomBuilder* pb = new MINCPhantomBuilder("brain/1mm/phantom.yaml");
    // phantom = pb->GetPhantom();
    // Phantom::Save("test1", phantom);

    // IPhantomBuilder* pb = new SheppLoganBuilder("shepplogan300.dat", 300, 1.0);
    // phantom = pb->GetPhantom();
    // Phantom::Save("shepplogan300", phantom);
    
    this->phantom = phantom;
    this->kern = kernel;
    this->seq = sequence;

    frame = &setup->GetFrame();
    mouse = &setup->GetMouse();
    engine = &setup->GetEngine();
    SetupPlugins();
    LoadResources();

    SetupCanvas();
    SetupWall();
    SetupOpenCL();
    SetupRF();
    SetupSim();
 
    QApplication *app = env->GetApplication();
    //app->setStyle("plastique");
    //app->setStyle("motif");
    app->setStyle("clearlooks");

    QGLWidget* glw = env->GetGLWidget();
    glw->setMaximumSize(glw->minimumSize());
    ui = new Ui::MRIUI();
    ui->setupUi(this);
    ui->glLayout->addWidget(glw);

    QObject::connect(ui->radioSim, SIGNAL(toggled(bool)),
                     this, SLOT(SetSimView(bool)));

    QObject::connect(ui->radioRF, SIGNAL(toggled(bool)),
                     this, SLOT(SetRFView(bool)));
    
    Slicer slicer(spinCanvas, fft);
    slicer.slices.push_back(phantomCanvas->GetSliceCanvas());
    // slicer.slices.push_back(samplesCanvas);
    // slicer.slices.push_back(fftCanvas);
    ValueList vl = slicer.Inspect();

    ActionValue* av = new ActionValueCall<MRIUI>(*this, &MRIUI::Exit);
    av->name = "Exit";
    vl.push_back(av);
    InspectionWidget *iw = new InspectionWidget("Slice",vl) ;
    iw->setMinimumWidth(200);
    setup->GetEngine().ProcessEvent().Attach(*iw);

    InspectionWidget *iw1 = new InspectionWidget("MRISim", sim->Inspect());
    iw1->setMinimumWidth(300);
    setup->GetEngine().ProcessEvent().Attach(*iw1);

    InspectionWidget *iw2 = new InspectionWidget("Sequence", seq->Inspect());
    iw2->setMinimumWidth(300);
    setup->GetEngine().ProcessEvent().Attach(*iw2);

    Flipper flipper(kern, spinCanvas);
    InspectionWidget *iw3 = new InspectionWidget("Flipper", flipper.Inspect());
    iw3->setMinimumWidth(300);
    setup->GetEngine().ProcessEvent().Attach(*iw3);



    dwI  = new QDockWidget("Slicer",this);
    dwI1 = new QDockWidget("Simulator ",this);
    dwI2 = new QDockWidget("Sequence ",this);
    dwI3 = new QDockWidget("Kernel",this);

    dwI->setWidget(iw);
    dwI1->setWidget(iw1);
    dwI2->setWidget(iw2);
    dwI3->setWidget(iw3);

    SetSimView(true);
    SetRFView(false);
    show();
    setup->GetEngine().Start();
}


void MRIUI::SetSimView(bool toggle) {
    if (toggle) {
        mouse->MouseMovedEvent().Attach(*wcSim);
        mouse->MouseButtonEvent().Attach(*wcSim);
        cSwitch->SetCanvas(wcSim);

        addDockWidget(Qt::RightDockWidgetArea, dwI);
        addDockWidget(Qt::RightDockWidgetArea, dwI1);
        addDockWidget(Qt::RightDockWidgetArea, dwI2);
        addDockWidget(Qt::RightDockWidgetArea, dwI3);
        
        dwI->show();
        dwI1->show();
        dwI2->show();
        dwI3->show();
      
        ui->menuView->addAction(dwI->toggleViewAction());
        ui->menuView->addAction(dwI1->toggleViewAction());
        ui->menuView->addAction(dwI2->toggleViewAction());
        ui->menuView->addAction(dwI3->toggleViewAction());
    }
    else {
        removeDockWidget(dwI);
        removeDockWidget(dwI1);
        removeDockWidget(dwI2);
        removeDockWidget(dwI3);

        ui->menuView->removeAction(dwI->toggleViewAction());
        ui->menuView->removeAction(dwI1->toggleViewAction());
        ui->menuView->removeAction(dwI2->toggleViewAction());
        ui->menuView->removeAction(dwI3->toggleViewAction());

        mouse->MouseMovedEvent().Detach(*wcSim);
        mouse->MouseButtonEvent().Detach(*wcSim);
    }
}

void MRIUI::SetRFView(bool toggle) {
    if (toggle) {
        mouse->MouseMovedEvent().Attach(*wcRF);
        mouse->MouseButtonEvent().Attach(*wcRF);
        cSwitch->SetCanvas(wcRF);

        addDockWidget(Qt::RightDockWidgetArea, dwI4);
        addDockWidget(Qt::RightDockWidgetArea, dwI5);
        ui->menuView->addAction(dwI4->toggleViewAction());
        ui->menuView->addAction(dwI5->toggleViewAction());
        dwI4->show();
        dwI5->show();
    }
    else {
        removeDockWidget(dwI4);
        removeDockWidget(dwI5);
        ui->menuView->removeAction(dwI4->toggleViewAction());
        ui->menuView->removeAction(dwI5->toggleViewAction());

        mouse->MouseMovedEvent().Detach(*wcRF);
        mouse->MouseButtonEvent().Detach(*wcRF);
    }
}


void nop(MRIUI *n) {}

int main(int argc, char* argv[]) {
    
    QtEnvironment* env = new QtEnvironment(false, 650, 700, 32, 
                                           FrameOption(), argc, argv);
    SimpleSetup* setup = new SimpleSetup("MRISIM", env); // placed here to enable logging


    DirectoryManager::AppendPath("projects/MRISIM/data/");
    //ResourceManager<IFontResource>::AddPlugin(new CairoFontPlugin());
    ResourceManager<MINCResource>::AddPlugin(new MINCPlugin());


    MRICommandLine cmdl(argc, argv);
    IMRISequence* sequence = cmdl.GetSequence();
    Phantom phantom = cmdl.GetPhantom();
    IMRIKernel* kernel = cmdl.GetKernel();

    // --- box phantom ---
    //IPhantomBuilder* pb = new SimplePhantomBuilder(phantomSize);
    //Phantom p = pb->GetPhantom();
    // -- phantom loaded from yaml file ---
    // Phantom p = Phantom("test.yaml");
    
    MRIUI *ui = new MRIUI(env, sequence, kernel, phantom, setup);
    nop(ui);   
}
