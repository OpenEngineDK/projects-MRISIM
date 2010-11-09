// main
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

// OpenEngine stuff
#include <Meta/Config.h>
#include <Logging/Logger.h>
#include <Logging/StreamLogger.h>
#include <Core/Engine.h>
#include <Resources/ITexture2D.h>
#include <Resources/ResourceManager.h>
#include <Resources/IFontResource.h>
#include <Scene/TransformationNode.h>
#include <Scene/SceneNode.h>
#include <Geometry/FaceSet.h>
#include <Renderers/TextureLoader.h>
//#include <Utils/CairoTextTool.h>
#include <Science/Plot.h>
#include <Science/MathGLPlot.h>
#include <Science/PointGraphDataSet.h>

//#include <Resources/CairoFont.h>
#include <Resources/SDLFont.h>

// SimpleSetup
#include <Utils/SimpleSetup.h>
#include <Display/ICanvas.h>
#include <Display/CanvasQueue.h>
#include <Display/OpenGL/SplitScreenCanvas.h>
#include <Display/OpenGL/TextureCopy.h>

#include <Display/WallCanvas.h>
#include <Display/GridLayout.h>
#include <Display/IFrame.h>
#include <Renderers/OpenGL/Renderer.h>

#include <Display/AntTweakBar.h>
#include <Utils/InspectionBar.h>
#include <Utils/IInspector.h>

// medical data loader
#include "Resources/MINCResource.h"
#include "Display/OpenGL/SliceCanvas.h"
#include "Display/OpenGL/PhantomCanvas.h"

#include "Resources/Phantom.h"
#include "Resources/SimplePhantomBuilder.h"

#include <Utils/PropertyTree.h>

// name spaces that we will be using.
// this combined with the above imports is almost the same as
// fx. import OpenEngine.Logging.*; in Java.
using namespace OpenEngine::Logging;
using namespace OpenEngine::Core;
using namespace OpenEngine::Utils;
using namespace OpenEngine::Resources;
using namespace OpenEngine::Scene;
using namespace OpenEngine::Geometry;
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Renderers::OpenGL;
using namespace OpenEngine::Science;
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
        values.push_back(v);
    }
    return values;
    
}

}}}


// class Delayed3dTextureLoader 
//     : public IListener<Renderers::RenderingEventArg> {
// private:
//     ITexture3DPtr tex;
// public:
//     Delayed3dTextureLoader(ITexture3DPtr tex) : tex(tex) {}
//     void Handle(RenderingEventArg arg) {
//         arg.renderer.LoadTexture(tex);
//     }
// };

/**
 * Main method for the first quarter project of CGD.
 * Corresponds to the
 *   public static void main(String args[])
 * method in Java.
 */
int main(int argc, char** argv) {
    SimpleSetup* setup = new SimpleSetup("MRISIM"); 
    logger.info << "========= Running OpenEngine Test Project =========" << logger.end;


    DirectoryManager::AppendPath("projects/MRISIM/data/");
    //ResourceManager<IFontResource>::AddPlugin(new CairoFontPlugin());
    ResourceManager<IFontResource>::AddPlugin(new SDLFontPlugin());
    ResourceManager<MINCResource>::AddPlugin(new MINCPlugin());
    
    IPhantomBuilder* pb = new SimplePhantomBuilder();
    Phantom p = pb->GetPhantom();
    // Phantom::Save("test", p);

    // PropertyTree ptree("brain.yaml");
    //PropertyTree ptree("test.yaml");
    // Phantom p(ptree);

    PhantomCanvas* pc = new PhantomCanvas(new TextureCopy(), p);

    IRenderer* r = new Renderer();
    // TextureLoader& tl = setup->GetTextureLoader(); 
    TextureLoader* tl = new TextureLoader(*r);
    
    
    // load medical data
    // MINCResourcePtr phantom = ResourceManager<MINCResource>::Create("brain/2/phantom_1.0mm_normal_gry.mnc"); 
    // MINCResourcePtr phantom = ResourceManager<MINCResource>::Create("brain/2/test.mnc");
   // MINCResourcePtr phantom = ResourceManager<MINCResource>::Create("brain/2/phantom_1.0mm_normal_csf.mnc");
    // phantom->Load();
 
    // SliceCanvas* sc = new SliceCanvas(new TextureCopy(), phantom->GetTexture3D());    

   
    IFontResourcePtr font = ResourceManager<IFontResource>::Create("Fonts/FreeSerifBold.ttf");
    font->Load();
    font->SetSize(24);
    font->SetColor(Vector<3,float>(1,0,0));

    //Wall wall(setup->GetTextureLoader(), font);

    AntTweakBar *atb = new AntTweakBar();
    atb->AttachTo(*r);

    ITweakBar *bar = new InspectionBar("minc",OpenEngine::Utils::Inspection::Inspect(pc->GetSliceCanvas()));     
    atb->AddBar(bar);
    bar->SetPosition(Vector<2,float>(20,40));
    bar->SetIconify(false);
    
    WallCanvas *wc = new WallCanvas(new TextureCopy(), *r, *tl, font, new GridLayout());

    setup->GetKeyboard().KeyEvent().Attach(*atb);
    setup->GetMouse().MouseMovedEvent().Attach(*atb);
    setup->GetMouse().MouseButtonEvent().Attach(*atb);

    atb->MouseMovedEvent().Attach(*wc);
    atb->MouseButtonEvent().Attach(*wc);

    ICanvas *mainC = setup->GetCanvas();
    IFrame& frame = setup->GetFrame();
    // ICanvas *splitCanvas = new SplitScreenCanvas<TextureCopy>(*mainC, *wc);

    CanvasQueue* cq = new CanvasQueue();
    cq->PushCanvas(wc);
    cq->PushCanvas(pc);
    //frame.SetCanvas(splitCanvas);
    //frame.SetCanvas(wc);
    frame.SetCanvas(cq);

    // ITextureResourcePtr trans = phantom->CreateTransverseSlice(50);
    // trans->Load();
    // tl.Load(trans);
    // wc->AddTextureWithText(trans, "Transverse");

    // ITextureResourcePtr sag = phantom->CreateSagitalSlice(50);    
    // sag->Load();
    // tl.Load(sag, TextureLoader::RELOAD_QUEUED);
    // wc->AddTextureWithText(sag, "Sagital");

    // ITextureResourcePtr cor = phantom->CreateCoronalSlice(50);
    // cor->Load();
    // tl.Load(cor);
    // wc->AddTextureWithText(cor, "Coronal");

    wc->AddTextureWithText(pc->GetTexture(), "phantom");

    Plot* plot = new Plot(Vector<2,float>(0, 100),
                          Vector<2,float>(0, 1));
    PointGraphDataSet *data1 = new PointGraphDataSet(100, 0, 100);;

    plot->AddDataSet(data1);
    
    for (int i=0;i<100;i++) {
        float y = float(i)/100.0;
        //logger.info << y << logger.end;
        data1->SetValue(i,y);
    }

    EmptyTextureResourcePtr plotTex = EmptyTextureResource::Create(200,200,24);
    plot->RenderInEmptyTexture(plotTex);
    tl->Load(plotTex);
    wc->AddTextureWithText(plotTex, "plot");

    MathGLPlot *plot2 = new MathGLPlot(400,400);
    tl->Load(plot2->GetTexture());
    
    wc->AddTextureWithText(plot2->GetTexture(), "mathgl");

    //ISceneNode* wallNode = wall.MakeScene();
    
    
    

    //    setup->SetScene(*wallNode);

    float h = -25/2;
    setup->GetCamera()->SetPosition(Vector<3,float>(0.0,h,80));
    setup->GetCamera()->LookAt(Vector<3,float>(0.0,h,0.0));



    // Start the engine.
    setup->GetEngine().Start();

    // Return when the engine stops.
    return EXIT_SUCCESS;
}


