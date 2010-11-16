#include "BlochTest.h"
#include <Logging/Logger.h>

namespace MRI {
namespace Science {

using namespace Scene;

BlochTest::BlochTest() {
    timeScale = 0.0001;
    B_0 = Vector<3,float>(0,0,1); // tesla
    M = Vector<3,float>(0.01);
    gyro = 42576; // Mhz/Tesla
    T_1 = 240/1000.0;
    T_2 = 70/1000.0;
}

void BlochTest::SetNode(SpinNode *sn) {
    spinNode = sn;
}    

void BlochTest::Handle(Core::ProcessEventArg arg) {
    float secs = arg.approx/1000000.0;
    float dt = secs*timeScale;
    time += dt;
    
    float M_0 = 0;

    // presession
    Vector<3,float> MDT = gyro*(M % B_0);    

    float Mdt_x = MDT[0] - M[0]/T_2;
    float Mdt_y = MDT[1] - M[1]/T_2;
    float Mdt_z = MDT[2] - (M[2] - M_0)/T_1;

    M += dt*Vector<3,float>(Mdt_x,
                            Mdt_y,
                            Mdt_z);

    //logger.info << M << logger.end;

    spinNode->M = M;
}

ValueList BlochTest::Inspect() {
    ValueList values;
    {
        RWValueCall<BlochTest, float > *v
            = new RWValueCall<BlochTest, float >(*this,
                                                 &BlochTest::GetTimeScale,
                                                 &BlochTest::SetTimeScale);
        v->name = "time scale";
        v->properties[STEP] = 0.0001;
        v->properties[MIN] = 0.0001;
        v->properties[MAX] = 0.01;
        values.push_back(v);
    }
    return values;

}

}
}

