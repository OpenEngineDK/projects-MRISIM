#ifndef _MRI_SPIN_NODE_H_
#define _MRI_SPIN_NODE_H_

#include <Scene/RenderNode.h>
#include <Math/Vector.h>
#include <Utils/Timer.h>
#include <vector>

namespace MRI {
namespace Scene {

using namespace OpenEngine;
using namespace OpenEngine::Scene;
using namespace OpenEngine::Math;
using namespace OpenEngine::Utils;

class SpinNode : public RenderNode {
    float scale;
    std::vector<Vector<3,float> > trace;
    int trace_idx;
    Timer traceTimer;
public:
    Vector<3,float> M;
    SpinNode();
    void Apply(Renderers::RenderingEventArg arg, ISceneNodeVisitor& v);
};

}
}

#endif
