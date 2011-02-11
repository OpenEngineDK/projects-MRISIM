// 3D texture based on complex cartesian grid based samples.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _MRI_SAMPLE_3D_TEXTURE_H_
#define _MRI_SAMPLE_3D_TEXTURE_H_

#include "../Science/MRISim.h"

#include <Math/Vector.h>
#include <Resources/Texture3D.h>
#include <Core/IListener.h>

#include <vector>
#include <complex>

namespace MRI {
namespace Resources {

using Science::SamplesChangedEventArg;
using OpenEngine::Math::Vector;
using OpenEngine::Resources::FloatTexture3DPtr;
using OpenEngine::Resources::FloatTexture3D;
using OpenEngine::Core::IListener;

using std::vector;
using std::complex;


/**
 * 3D textures based on complex cartesian grid based samples.
 * 
 * @class Sample3DTexture Sample3DTexture.h MRISIM/Science/Sample3DTexture.h
 */
class Sample3DTexture : public FloatTexture3D, public IListener<SamplesChangedEventArg> {
private:
    vector<complex<float> >& samples;
    FloatTexture3DPtr ref;
public:   
    Sample3DTexture(vector<complex<float> >& samples, Vector<3,unsigned int> dims);
    virtual ~Sample3DTexture();

    void Handle(SamplesChangedEventArg arg);
};

typedef boost::shared_ptr<Sample3DTexture > Sample3DTexturePtr;

    
} // NS Science
} // NS MRI

#endif //_MRI_SAMPLE_3D_TEXTURE_H_
