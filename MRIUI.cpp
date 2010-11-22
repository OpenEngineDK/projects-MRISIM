#include "MRIUI.h"
#include "MRISIM/ui_MRIUI.h"
// Global

#include <Meta/OpenGL.h>
#include <Meta/Config.h>


#include <Resources/DirectoryManager.h>
#include <Resources/ResourceManager.h>
#include <Resources/SDLFont.h>

#include <Display/SceneGraphGUI.h>
#include <Display/SceneNodeGUI.h>
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

#include "Scene/SpinNode.h"

#include "Science/BlochTest.h"
#include "Science/MRISim.h"
#include "Science/CPUKernel.h"

#include "Display/OpenGL/SpinCanvas.h"

#undef main // Evil hack :/

using namespace OpenEngine::Logging;
using namespace OpenEngine::Core;
using namespace OpenEngine::Utils;
using namespace OpenEngine::Resources;
using namespace OpenEngine::Scene;
//using namespace OpenEngine::Geometry;
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Renderers::OpenGL;
//using namespace OpenEngine::Science;
using namespace OpenEngine::Display;
using namespace OpenEngine::Display::OpenGL;

using namespace MRI::Scene;
using namespace MRI::Science;
using namespace MRI::Display::OpenGL;


namespace OpenEngine {
    namespace Utils {
        namespace Inspection {

ValueList Inspect(SliceCanvas* sc) {
    ValueList values;
    {
        RWValueCall<SliceCanvas, unsigned int > *v
            = new RWValueCall<SliceCanvas, unsigned int >(*sc,
                                                          &SliceCanvas::GetSlice,
                                                          &SliceCanvas::SetSlice);
        v->name = "slice";
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

void MRIUI::SetupCanvas() {
    IRenderer* r = new Renderer();
    TextureLoader* tl = new TextureLoader(*r);
    r->PreProcessEvent().Attach(*tl);

    wc = new WallCanvas(new TextureCopy(), *r, *tl, font, new GridLayout());


    mouse->MouseMovedEvent().Attach(*wc);
    mouse->MouseButtonEvent().Attach(*wc);


    sliceCanvas = new SliceCanvas(new TextureCopy(), phantom->GetTexture3D());    


    //wc->AddTextureWithText(sliceCanvas->GetTexture(), "hest");

    
    // IPhantomBuilder* pb = new SimplePhantomBuilder();
    // Phantom p = pb->GetPhantom();
    // Phantom::Save("test", p);

    // IPhantomBuilder* pb = new MINCPhantomBuilder("brain/2/phantom.yaml");
    // Phantom p = pb->GetPhantom();
    // Phantom::Save("test", p);

    Phantom p("brain.yaml");
    phantomCanvas = new PhantomCanvas(new TextureCopy(), p);

    plot = new MathGLPlot(400,200);
    fftPlot = new MathGLPlot(400,200);
    tl->Load(plot->GetTexture(), TextureLoader::RELOAD_IMMEDIATE);
    tl->Load(fftPlot->GetTexture(), TextureLoader::RELOAD_IMMEDIATE);

    wc->AddTextureWithText(plot->GetTexture(), "plot");

    wc->AddTextureWithText(fftPlot->GetTexture(), "fft");

    RenderCanvas *rc = new RenderCanvas(new TextureCopy(),Vector<2,int>(400,400));
    
    r = new Renderer();
    r->SetBackgroundColor(Vector<4,float>(0,0,0,1));
    RenderingView *rv = new RenderingView();

    r->ProcessEvent().Attach(*rv);
    r->InitializeEvent().Attach(*rv);
    

    rc->SetRenderer(r);
    Camera* cam = new Camera(*(new PerspectiveViewingVolume()));
    cam->SetPosition(Vector<3,float>(10,10,10));
    cam->LookAt(Vector<3,float>(0,0,0));
    rc->SetViewingVolume(cam);

    SpinNode *sn = new SpinNode();
    rc->SetScene(sn);

    spinNode = sn;

    wc->AddTextureWithText(rc->GetTexture(), "render");

    cq = new CanvasQueue();
    cq->PushCanvas(wc);
    cq->PushCanvas(sliceCanvas);
    cq->PushCanvas(phantomCanvas);
    cq->PushCanvas(rc);

    

    frame->SetCanvas(cq);


}

void MRIUI::LoadResources() {
    phantom = ResourceManager<MINCResource>::Create("brain/2/phantom_1.0mm_normal_gry.mnc");
    phantom->Load();
    font = ResourceManager<IFontResource>::Create("Fonts/FreeSansBold.ttf");
    font->Load();
    font->SetSize(24);
    font->SetColor(Vector<3,float>(1,0,0));

}

void MRIUI::Exit() {
    exit(0);
}

MRIUI::MRIUI(QtEnvironment *env) {
    SimpleSetup* setup = new SimpleSetup("MRISIM",env);
    frame = &setup->GetFrame();
    mouse = &setup->GetMouse();

    SetupPlugins();
    LoadResources();


    SetupCanvas();
    
    //QApplication *app = env->GetApplication();
    //app->setStyle("plastique");
    //app->setStyle("motif");

    // BlochTest *bt = new BlochTest();
    // bt->SetNode(spinNode);
    // setup->GetEngine().ProcessEvent().Attach(*bt);

    
    // test cpu simulator
    
    IPhantomBuilder* pb = new SimplePhantomBuilder();
    Phantom p = pb->GetPhantom();
    // Phantom::Save("test", p);

    CPUKernel* kern = new CPUKernel();
    MRISim* sim = new MRISim(p, kern);
    setup->GetEngine().InitializeEvent().Attach(*sim);
    setup->GetEngine().ProcessEvent().Attach(*sim);
    setup->GetEngine().DeinitializeEvent().Attach(*sim);
    sim->SetNode(spinNode);
    sim->SetPlot(plot);
    sim->SetFFTPlot(fftPlot);
    sim->Start();

 
    SpinCanvas* sc = new SpinCanvas(new TextureCopy(), *kern, setup->GetRenderer(), 200, 200);
    cq->PushCanvas(sc);
    wc->AddTextureWithText(sc->GetTexture(), "spins");

    ui = new Ui::MRIUI();
    ui->setupUi(this);

    ui->topLayout->addWidget(env->GetGLWidget());
    SelectionSet<ISceneNode>* ss = new SelectionSet<ISceneNode>(); 
    SceneGraphGUI *graphGui = new SceneGraphGUI(spinNode,
                                                &setup->GetTextureLoader(), *ss);
    SceneNodeGUI *nodeGui = new SceneNodeGUI();

    // inspector
    ValueList vl = Inspection::Inspect(sliceCanvas);
    ValueList vl2 = Inspection::Inspect(phantomCanvas->GetSliceCanvas());
    vl.merge(vl2);
    

    ActionValue* av = new ActionValueCall<MRIUI>(*this, &MRIUI::Exit);
    av->name = "Exit";
    vl.push_back(av);
    InspectionWidget *iw = new InspectionWidget("Slice",vl) ;
    iw->setMinimumWidth(200);
    setup->GetEngine().ProcessEvent().Attach(*iw);

    // InspectionWidget *iw2 = new InspectionWidget("Bloch",bt->Inspect());
    InspectionWidget *iw2 = new InspectionWidget("MRISim",sim->Inspect());
    iw2->setMinimumWidth(300);
    setup->GetEngine().ProcessEvent().Attach(*iw2);

    InspectionWidget *iw3 = new InspectionWidget("MRISim",kern->Inspect());
    iw3->setMinimumWidth(300);
    setup->GetEngine().ProcessEvent().Attach(*iw3);

    QDockWidget* dwSG = new QDockWidget("Scene Graph",this);
    QDockWidget* dwSN = new QDockWidget("Scene Node",this);
    QDockWidget *dwI  = new QDockWidget("Slice Inspector",this);
    QDockWidget *dwI2 = new QDockWidget("MRISim Inspector",this);
    QDockWidget *dwI3 = new QDockWidget("Kernel Inspector",this);

    dwSG->setWidget(graphGui);
    dwSN->setWidget(nodeGui);
    dwI->setWidget(iw);
    dwI2->setWidget(iw2);
    dwI3->setWidget(iw3);
    
    addDockWidget(Qt::RightDockWidgetArea, dwSG);
    dwSG->close();
    addDockWidget(Qt::RightDockWidgetArea, dwSN);
    dwSN->close();
    addDockWidget(Qt::RightDockWidgetArea, dwI);
    addDockWidget(Qt::RightDockWidgetArea, dwI2);
    addDockWidget(Qt::RightDockWidgetArea, dwI3);

    ui->menuView->addAction(dwSG->toggleViewAction());
    ui->menuView->addAction(dwSN->toggleViewAction());
    ui->menuView->addAction(dwI->toggleViewAction());
    ui->menuView->addAction(dwI2->toggleViewAction());
    ui->menuView->addAction(dwI3->toggleViewAction());
    
    graphGui->SelectionEvent().Attach(*nodeGui);
    graphGui->SelectionEvent().Attach(*graphGui);


    setup->GetEngine().InitializeEvent().Attach(*graphGui);
    
    show();
    setup->GetEngine().Start();
}



int main(int argc, char* argv[]) {
    QtEnvironment* env = new QtEnvironment(false, 800, 600, 32, 
                                           FrameOption(), argc, argv);
    MRIUI *ui = new MRIUI(env);
}
