#include "SpinNode.h"

#include <Logging/Logger.h>

namespace MRI {
namespace Scene {

using namespace OpenEngine::Renderers;

SpinNode::SpinNode() {
    scale = 10;
}

void SpinNode::Apply(Renderers::RenderingEventArg arg, ISceneNodeVisitor& v) {
    IRenderer& rend = arg.renderer;
    Vector<3,float> zero(0,0,0);
    
    Line xaxis(zero, Vector<3,float>(1,0,0)*scale);
    Line yaxis(zero, Vector<3,float>(0,1,0)*scale);
    Line zaxis(zero, Vector<3,float>(0,0,1)*scale);

    rend.DrawLine(xaxis, Vector<3,float>(1,0,0));
    rend.DrawLine(yaxis, Vector<3,float>(0,1,0));
    rend.DrawLine(zaxis, Vector<3,float>(0,0,1));



    Line l(zero,
           M*scale);
    rend.DrawLine(l, Vector<3,float>(1,1,1));
}

}
}
