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
#include <Scene/TransformationNode.h>
#include <Scene/SceneNode.h>
#include <Geometry/FaceSet.h>
#include <Renderers/TextureLoader.h>
#include <Utils/CairoTextTool.h>
#include <Science/Plot.h>
#include <Science/MathGLPlot.h>
#include <Science/PointGraphDataSet.h>

// SimpleSetup
#include <Utils/SimpleSetup.h>

// Game factory
//#include "GameFactory.h"

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

// Helpers
static TransformationNode* CreateTextureBillboard(ITextureResourcePtr texture,
                                                  float scale) {
    unsigned int textureHosisontalSize = texture->GetWidth();
    unsigned int textureVerticalSize = texture->GetHeight();

    logger.info << "w x h = " << texture->GetWidth()
                << " x " << texture->GetHeight() << logger.end;
    float fullxtexcoord = 1;
    float fullytexcoord = 1;
  
    FaceSet* faces = new FaceSet();

    float horisontalhalfsize = textureHosisontalSize * 0.5;
    Vector<3,float> lowerleft = Vector<3,float>(horisontalhalfsize,0,0);
    Vector<3,float> lowerright = Vector<3,float>(-horisontalhalfsize,0,0);
    Vector<3,float> upperleft = Vector<3,float>(horisontalhalfsize,textureVerticalSize,0);
    Vector<3,float> upperright = Vector<3,float>(-horisontalhalfsize,textureVerticalSize,0);

    FacePtr leftside = FacePtr(new Face(lowerleft,lowerright,upperleft));

    /*
      leftside->texc[1] = Vector<2,float>(1,0);
      leftside->texc[0] = Vector<2,float>(0,0);
      leftside->texc[2] = Vector<2,float>(0,1);
    */
    leftside->texc[1] = Vector<2,float>(0,fullytexcoord);
    leftside->texc[0] = Vector<2,float>(fullxtexcoord,fullytexcoord);
    leftside->texc[2] = Vector<2,float>(fullxtexcoord,0);
    leftside->norm[0] = leftside->norm[1] = leftside->norm[2] = Vector<3,float>(0,0,1);
    leftside->CalcHardNorm();
    leftside->Scale(scale);
    faces->Add(leftside);

    FacePtr rightside = FacePtr(new Face(lowerright,upperright,upperleft));
    /*
      rightside->texc[2] = Vector<2,float>(0,1);
      rightside->texc[1] = Vector<2,float>(1,1);
      rightside->texc[0] = Vector<2,float>(1,0);
    */
    rightside->texc[2] = Vector<2,float>(fullxtexcoord,0);
    rightside->texc[1] = Vector<2,float>(0,0);
    rightside->texc[0] = Vector<2,float>(0,fullytexcoord);
    rightside->norm[0] = rightside->norm[1] = rightside->norm[2] = Vector<3,float>(0,0,1);
    rightside->CalcHardNorm();
    rightside->Scale(scale);
    faces->Add(rightside);

    MaterialPtr m = leftside->mat = rightside->mat = MaterialPtr(new Material());
    m->AddTexture(texture);

    GeometryNode* node = new GeometryNode();
    node->SetFaceSet(faces);
    TransformationNode* tnode = new TransformationNode();
    tnode->AddNode(node);
    return tnode;
}

struct WallItem {
    ITextureResourcePtr texture;
    string title;
    Vector<2,unsigned int> scale;

    WallItem() {}

    WallItem(ITextureResourcePtr t, 
             string s)
        : texture(t)
        , title(s)
        , scale(Vector<2,unsigned int>(1,1)) {
        
    }
};

struct Wall {
    WallItem tex[12];
    TextureLoader& loader;

    Wall(TextureLoader& l) : loader(l) {
        
    }

    WallItem& operator()(int x, int y) {
        return tex[x*3+y];
    }
    ISceneNode* MakeScene() {
        SceneNode *sn = new SceneNode();
        CairoTextTool textTool;
        
        for (int x=0;x<4;x++) {
            for (int y=0;y<3;y++) {
                WallItem itm = (*this)(x,y);
                ITextureResourcePtr t = itm.texture;
                if (t) {
                    Vector<2,unsigned int> scale = itm.scale;
                    loader.Load(t,TextureLoader::RELOAD_QUEUED);
                    TransformationNode* node = new TransformationNode();
                    TransformationNode* bnode = CreateTextureBillboard(t,0.05);
                    bnode->SetScale(Vector<3,float>( 1.0 * scale[0],
                                                   -1.0 * scale[1],
                                                    1.0));
                    node->Move(x*35-25,y*35-25,0);
                    node->AddNode(bnode);
                    
                    CairoResourcePtr textRes = CairoResource::Create(128,32);
                    textRes->Load();

                    ostringstream out;
                    out << "(" << x << "," << y << ") " << itm.title;

                    textTool.DrawText(out.str(), textRes);

                    loader.Load(textRes);
                    TransformationNode* textNode = CreateTextureBillboard(textRes,0.15);

                    textNode->Move(0,-28,-0.01);


                    node->AddNode(textNode);
                    //sn->AddNode(textNode);
                    sn->AddNode(node);
                }
            }
        }

        return sn;
    }
};


/**
 * Main method for the first quarter project of CGD.
 * Corresponds to the
 *   public static void main(String args[])
 * method in Java.
 */
int main(int argc, char** argv) {
    // Setup logging facilities.
    Logger::AddLogger(new StreamLogger(&std::cout));

    // Print usage info.
    logger.info << "========= Running OpenEngine Test Project =========" << logger.end;

    // Create simple setup
    SimpleSetup* setup = new SimpleSetup("MRISIM");

    Wall wall(setup->GetTextureLoader());
    
    ITextureResourcePtr img = ResourceManager<ITextureResource>::Create("test.png");
    img->Load();

    wall(0,0) = WallItem(img, "test");

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

    wall(0,1) = WallItem(plotTex, "plot");

    MathGLPlot *plot2 = new MathGLPlot(400,400);
    wall(1,1) = WallItem(plot2->GetTexture(), "mathgl");
    

    ISceneNode* wallNode = wall.MakeScene();

    setup->SetScene(*wallNode);

    float h = -25/2;
    setup->GetCamera()->SetPosition(Vector<3,float>(0.0,h,80));
    setup->GetCamera()->LookAt(Vector<3,float>(0.0,h,0.0));



    // Start the engine.
    setup->GetEngine().Start();

    // Return when the engine stops.
    return EXIT_SUCCESS;
}


