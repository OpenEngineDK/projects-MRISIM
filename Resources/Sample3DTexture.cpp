// 3D texture based on complex cartesian grid based samples.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#include "Sample3DTexture.h"

namespace MRI {
namespace Resources {

using namespace OpenEngine::Math;
using namespace OpenEngine::Resources;
using namespace std;

Sample3DTexture::Sample3DTexture(vector<complex<float> >& samples, Vector<3,unsigned int> dims)
    : FloatTexture3D(dims[0] > 0 ? dims[0] : 1, dims[1] > 0 ? dims[1] : 1, dims[2] > 0 ? dims[2] : 1, 1)
    , samples(samples)
    , ref(this)
{
    SetWrapping(CLAMP);
    SetFiltering(NONE);
    Handle(SamplesChangedEventArg(0, dims[0] * dims[1] * dims[2]));
}

Sample3DTexture::~Sample3DTexture() {
    ref.reset();
}

void Sample3DTexture::Handle(SamplesChangedEventArg arg) {
    for (unsigned int i = arg.begin; i < arg.end; ++i)
        ((float*)data)[i] = abs(samples[i]);
    this->ChangedEvent().Notify(Texture3DChangedEventArg(ref));
}

} // NS Science
} // NS MRI

