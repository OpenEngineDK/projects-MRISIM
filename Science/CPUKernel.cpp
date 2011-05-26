// MRI Simulator: cpu kernel implementation
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "CPUKernel.h"

#include <Math/Matrix.h>
#include <Logging/Logger.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils::Inspection;
using namespace OpenEngine::Math;

CPUKernel::CPUKernel() 
    : refMagnets(NULL)
    , labMagnets(NULL)
    , eq(NULL)
    , deltaB0(NULL)
    , gradient(Vector<3,float>(0.0))
    , rfSignal(Vector<3,float>(0.0))
    , data(NULL)
    , width(0)
    , height(0)
    , depth(0)
    , sz(0)
    , b0(0.5)
    , gyro(GYRO_RAD) // radians/Tesla
    , time(0.0)
    , omega0Angle(0.0) {
    randomgen.SeedWithTime();
    // rn = new KernRenderNode(this);
}

CPUKernel::~CPUKernel() {
}

float CPUKernel::RandomAttribute(float base, float variance) {
    return base + randomgen.UniformFloat(-1.0,1.0) * variance;
}

void CPUKernel::Init(Phantom phantom) {
    this->phantom = phantom;
    width  = phantom.texr->GetWidth();
    height = phantom.texr->GetHeight();
    depth  = phantom.texr->GetDepth();
    sz = width*height*depth;
    refMagnets = new Vector<3,float>[sz]; 
    labMagnets = new Vector<3,float>[sz]; 
    eq = new float[sz];
    deltaB0 = new float[sz];
    data = phantom.texr->GetData();

    for (unsigned int i = 0; i < sz; ++i) {
        //deltaB0[i] = RandomAttribute(0.0, 1e-3);
        deltaB0[i] = 0.0;
        eq[i] = phantom.spinPackets[data[i]].ro*b0;
        // logger.info << "data[" << i << "] = " << int(data[i]) << logger.end;
        // logger.info << "eq[" << i << "] = " << eq[i] << logger.end;
    }
    Reset();
}

inline Vector<3,float> RotateX(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(1.0, 0.0, 0.0,
                        0.0, cos(angle), sin(angle),
                        0.0, -sin(angle), cos(angle));
    return m*vec;
}

inline Vector<3,float> RotateY(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(cos(angle), 0.0, sin(angle),
                               0.0, 1.0,        0.0,
                       -sin(angle), 0.0, cos(angle));
    return m*vec;
}

inline Vector<3,float> RotateZ(float angle, Vector<3,float> vec) {
    Matrix<3,3,float> m(cos(angle), -sin(angle), 0.0,
                        sin(angle), cos(angle), 0.0,
                        0.0, 0.0, 1.0);
    return m*vec;
}


void CPUKernel::Step(float dt) {
    time += dt;
    signal = Vector<3,float>();
    const double omega0 = GYRO_RAD * b0;
    // const double omega0Angle = fmod(omega0*time, double(Math::PI * 2.0));

    omega0Angle += dt * omega0;
    if (omega0Angle > double(Math::PI * 2.0))
        omega0Angle = fmod(omega0Angle, double(Math::PI * 2.0));

    // move rf signal into reference space
    const Vector<3,float> rf = RotateZ(-omega0Angle, rfSignal);
    // const Vector<3,float> rf = rfSignal;
    // logger.info << "RFSIGNAL: " << rf << logger.end; 
    // logger.info << "Bx: " << rf.Get(0) << logger.end;
    // logger.info << "By: " << rf.Get(1) << logger.end;
    
    //rn->magnet = rfSignal.GetNormalize();
    // rn->rf = rfSignal;

    // logger.info << "Ref " <<  refMagnets[0] << logger.end;

    for (unsigned int x = 0; x < width; ++x) {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int z = 0; z < depth; ++z) {

                unsigned int i = x + y*width + z*width*height;
                if (data[i] == 0) continue;
                

                float dtt1 = dt/phantom.spinPackets[data[i]].t1;
                float dtt2 = dt/phantom.spinPackets[data[i]].t2;

                refMagnets[i] += Vector<3,float>(-refMagnets[i][0]*dtt2, 
                                                 -refMagnets[i][1]*dtt2, 
                                                 (eq[i]-refMagnets[i][2])*dtt1);

                float dG = gradient * Vector<3,float>(float(int(x) + phantom.offsetX) * (phantom.sizeX*1e-3),
                                                      float(int(y) + phantom.offsetY) * (phantom.sizeY*1e-3),
                                                      float(int(z) + phantom.offsetZ) * (phantom.sizeZ*1e-3));



                refMagnets[i] = RotateZ(GYRO_RAD * (deltaB0[i] + dG) * dt, refMagnets[i]);

                refMagnets[i] = RotateX(rf.Get(0) * GYRO_RAD * dt, refMagnets[i]);
                refMagnets[i] = RotateY(rf.Get(1) * GYRO_RAD * dt, refMagnets[i]);
                
                labMagnets[i] = 
                    RotateZ(omega0Angle, refMagnets[i]);
                signal += refMagnets[i];
            }    
        }
    }
    //logger.error << refMagnets[500] << logger.end;


}

void CPUKernel::Flip(unsigned int slice) {
    RFPulse(Math::PI * 0.5, slice);
}

void CPUKernel::Flop(unsigned int slice) {
    RFPulse(Math::PI, slice);
}

Vector<3,float> CPUKernel::GetSignal() const {
    return signal;
}

void CPUKernel::SetB0(float b0) {
    this->b0 = b0;
}

float CPUKernel::GetB0() const {
    return b0;
}

void CPUKernel::RFPulse(float angle, unsigned int slice) {
    Matrix<3,3,float> rot(
                          1.0, 0.0, 0.0,
                          0.0, cos(angle), sin(angle),
                          0.0,-sin(angle), cos(angle)
                          );
    // hack to only excite slice 0
    unsigned int z = slice;
    for (unsigned int i = 0; i < width; ++i) {
        for (unsigned int j = 0; j < height; ++j) {
            if (data[i + j*width  + z*width*height] == 0) continue;
            refMagnets[i + j*width + z*width*height] = rot*refMagnets[i + j*width + z*width*height];
        }
    }
}

void CPUKernel::SetGradient(Vector<3,float> gradient) {
    this->gradient = gradient;
}

Vector<3,float> CPUKernel::GetGradient() const {
    return gradient;
}

void CPUKernel::SetRFSignal(Vector<3,float> signal) {
    rfSignal = signal;
}

void CPUKernel::Reset() {
    // initialize refMagnets to b0 * spin density 
    // Signal should at all times be the sum of the spins (or not?)
    omega0Angle = time = 0.0;
    signal = Vector<3,float>();
    for (unsigned int i = 0; i < sz; ++i) {
        refMagnets[i] = labMagnets[i] = Vector<3,float>(0.0, 0.0, eq[i]);
        signal += labMagnets[i];
        //signal += refMagnets[i];
    }
    logger.info << "Signal: " << signal << logger.end;
}

Vector<3,float>* CPUKernel::GetMagnets() const {
    //return refMagnets;
    return labMagnets;
}

Phantom CPUKernel::GetPhantom() const {
  return phantom;
}

// RenderNode* CPUKernel::GetRenderNode() {
//     return rn;
// }

// CPUKernel::KernRenderNode::KernRenderNode(CPUKernel *k) : kern(k) {

// }

// void CPUKernel::KernRenderNode::Apply(Renderers::RenderingEventArg arg, OpenEngine::Scene::ISceneNodeVisitor& v) {

//     using namespace OpenEngine::Renderers;

//     float scale = 10;

//     IRenderer& rend = arg.renderer;
//     Vector<3,float> zero(0,0,0);
    
//     Line xaxis(zero, Vector<3,float>(1,0,0)*scale);
//     Line yaxis(zero, Vector<3,float>(0,1,0)*scale);
//     Line zaxis(zero, Vector<3,float>(0,0,1)*scale);

//     rend.DrawLine(xaxis, Vector<3,float>(1,0,0));
//     rend.DrawLine(yaxis, Vector<3,float>(0,1,0));
//     rend.DrawLine(zaxis, Vector<3,float>(0,0,1));

//     Line l(zero, magnet*scale);
//     Line l2(magnet*scale, (magnet*scale+rf*400));
    
//     rend.DrawLine(l, Vector<3,float>(1,0,0),2);
//     rend.DrawLine(l2, Vector<3,float>(0,1,0),2);
    


// }

} // NS Science
} // NS OpenEngine
