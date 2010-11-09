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
#include <Display/CanvasQueue.h>
#include <Display/WallCanvas.h>
#include <Display/GridLayout.h>
#include <Display/InspectionWidget.h>

#include <Display/OpenGL/TextureCopy.h>

#include <Renderers/OpenGL/Renderer.h>

#include <Utils/SimpleSetup.h>
#include <Utils/IInspector.h>

// Local
#include "Resources/MINCResource.h"
#include "Resources/Phantom.h"
#include "Resources/SimplePhantomBuilder.h"
#include "Display/OpenGL/SliceCanvas.h"
#include "Display/OpenGL/PhantomCanvas.h"


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
    WallCanvas *wc = new WallCanvas(new TextureCopy(), *r, *tl, font, new GridLayout());


    mouse->MouseMovedEvent().Attach(*wc);
    mouse->MouseButtonEvent().Attach(*wc);


    sliceCanvas = new SliceCanvas(new TextureCopy(), phantom->GetTexture3D());    


    wc->AddTextureWithText(sliceCanvas->GetTexture(), "hest");



    
    IPhantomBuilder* pb = new SimplePhantomBuilder();
    Phantom p = pb->GetPhantom();
    // Phantom::Save("test", p);

    //PropertyTree ptree(DirectoryManager::FindFileInPath("brain.yaml"));
    //PropertyTree ptree("test.yaml");
    //Phantom p(ptree);

    phantomCanvas = new PhantomCanvas(new TextureCopy(), p);


    wc->AddTextureWithText(phantomCanvas->GetTexture(), "phantom");


    CanvasQueue *cq = new CanvasQueue();
    cq->PushCanvas(wc);
    cq->PushCanvas(sliceCanvas);
    cq->PushCanvas(phantomCanvas);

    

    frame->SetCanvas(cq);


}

void MRIUI::LoadResources() {
    phantom = ResourceManager<MINCResource>::Create("brain/2/phantom_1.0mm_normal_gry.mnc");
    phantom->Load();
    font = ResourceManager<IFontResource>::Create("Fonts/FreeSerifBold.ttf");
    font->Load();
    font->SetSize(24);
    font->SetColor(Vector<3,float>(1,0,0));

}

MRIUI::MRIUI(QtEnvironment *env) {
    SimpleSetup* setup = new SimpleSetup("MRISIM",env);
    frame = &setup->GetFrame();
    mouse = &setup->GetMouse();

    SetupPlugins();
    LoadResources();


    SetupCanvas();
    




    QApplication *app = env->GetApplication();
    //app->setStyle("cleanlooks");

    ui = new Ui::MRIUI();
    ui->setupUi(this);

    ui->topLayout->addWidget(env->GetGLWidget());
    SelectionSet<ISceneNode>* ss = new SelectionSet<ISceneNode>(); 
    SceneGraphGUI *graphGui = new SceneGraphGUI(setup->GetScene(), 
                                                &setup->GetTextureLoader(), *ss);
    SceneNodeGUI *nodeGui = new SceneNodeGUI();

    // inspector
    ValueList vl = Inspection::Inspect(sliceCanvas);
    ValueList vl2 = Inspection::Inspect(phantomCanvas->GetSliceCanvas());
    vl.merge(vl2);
    InspectionWidget *iw = new InspectionWidget("Slice",vl) ;
    setup->GetEngine().ProcessEvent().Attach(*iw);

    QDockWidget* dwSG = new QDockWidget("Scene Graph",this);
    QDockWidget* dwSN = new QDockWidget("Scene Node",this);
    QDockWidget *dwI = new QDockWidget("Slice Inspector",this);

    dwSG->setWidget(graphGui);
    dwSN->setWidget(nodeGui);
    dwI->setWidget(iw);
   
    addDockWidget(Qt::RightDockWidgetArea, dwSG);
    addDockWidget(Qt::RightDockWidgetArea, dwSN);
    addDockWidget(Qt::RightDockWidgetArea, dwI);

    ui->menuView->addAction(dwSG->toggleViewAction());
    ui->menuView->addAction(dwSN->toggleViewAction());
    ui->menuView->addAction(dwI->toggleViewAction());
    
    graphGui->SelectionEvent().Attach(*nodeGui);
    graphGui->SelectionEvent().Attach(*graphGui);


    setup->GetEngine().InitializeEvent().Attach(*graphGui);


    




    show();
    setup->GetEngine().Start();
}

int main(int argc, char* argv[]) {
    QtEnvironment* env = new QtEnvironment(false, 800, 600);
    MRIUI *ui = new MRIUI(env);
}
