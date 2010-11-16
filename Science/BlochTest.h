// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_BLOCH_TEST_H_
#define _OE_BLOCH_TEST_H_

#include <Core/EngineEvents.h>
#include <Core/IListener.h>

#include <Math/Vector.h>
#include <Utils/IInspector.h>


#include "../Scene/SpinNode.h"

namespace MRI {
namespace Science {

using namespace OpenEngine;
using namespace OpenEngine::Math;
using namespace OpenEngine::Core;
using namespace OpenEngine::Utils::Inspection;

class BlochTest : public IListener<Core::ProcessEventArg> {
    Scene::SpinNode* spinNode;
    float timeScale;
    float time;
    Vector<3,float> B_0;
    float gyro;
    Vector<3,float> M;
    float T_1;
    float T_2;
public:
    BlochTest();

    float GetTimeScale() { return timeScale; }
    void SetTimeScale(float s) { timeScale = s; }

    void SetNode(Scene::SpinNode *sn);
    void Handle(Core::ProcessEventArg arg);
    ValueList Inspect();
};


} // NS Science
} // NS OpenEngine

#endif // _OE_BLOCH_TEST_H_
