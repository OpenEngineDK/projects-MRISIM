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
#include <Display/OpenGL/SplitScreenCanvas.h>

#include <Display/WallCanvas.h>
#include <Display/IFrame.h>

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
using namespace OpenEngine::Science;
using namespace OpenEngine::Display;


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
    

    IFontResourcePtr font = ResourceManager<IFontResource>::Create("Fonts/FreeSerifBold.ttf");
    font->Load();
    font->SetSize(24);
    font->SetColor(Vector<3,float>(1,0,0));

    //Wall wall(setup->GetTextureLoader(), font);
    
    TextureLoader& tl = setup->GetTextureLoader(); 
    WallCanvas *wc = new WallCanvas(setup->GetRenderer(), tl, font);

    setup->GetMouse().MouseMovedEvent().Attach(*wc);
    setup->GetMouse().MouseButtonEvent().Attach(*wc);

    ICanvas *mainC = setup->GetCanvas();
    IFrame& frame = setup->GetFrame();
    ICanvas *splitCanvas = new SplitScreenCanvas(*mainC, *wc);

    //frame.SetCanvas(splitCanvas);
    frame.SetCanvas(wc);





    ITextureResourcePtr img = ResourceManager<ITextureResource>::Create("test.png");
    img->Load();
    tl.Load(img);

    wc->AddTextureWithText(img, "test");

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
    tl.Load(plotTex);
    
    wc->AddTextureWithText(plotTex, "plot");

    MathGLPlot *plot2 = new MathGLPlot(400,400);
    tl.Load(plot2->GetTexture());
    
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


