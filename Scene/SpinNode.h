#ifndef _MRI_SPIN_NODE_H_
#define _MRI_SPIN_NODE_H_

#include <Scene/RenderNode.h>
#include <Math/Vector.h>

namespace MRI {
namespace Scene {

using namespace OpenEngine;
using namespace OpenEngine::Scene;
using namespace OpenEngine::Math;

class SpinNode : public RenderNode {
    float scale;
public:
    Vector<3,float> M;
    SpinNode();
    void Apply(Renderers::RenderingEventArg arg, ISceneNodeVisitor& v);
};

}
}

#endif
