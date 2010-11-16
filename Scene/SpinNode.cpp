#include "SpinNode.h"

#include <Logging/Logger.h>

namespace MRI {
namespace Scene {

using namespace OpenEngine::Renderers;
using namespace std;

SpinNode::SpinNode() {
    scale = 10;
    trace = vector<Vector<3,float> >(10);
    traceTimer.Start();
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
    if (traceTimer.GetElapsedIntervals(100000)) {
        trace[trace_idx++ % 10] = l.point2;    
        
        traceTimer.Reset();
    }

    rend.DrawLine(l, Vector<3,float>(1,1,1),2);
    // tracer
    for (int i=0;i<10;i++) {
        l = Line(l.point2, trace[(trace_idx-1 - i) % 10]);
        rend.DrawLine(l, Vector<3,float>(1.0-i/10.0));
    }

        
    
}

}
}
