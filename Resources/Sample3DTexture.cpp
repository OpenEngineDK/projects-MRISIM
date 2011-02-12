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

Sample3DTexture::Sample3DTexture(vector<complex<float> >& samples, Vector<3,unsigned int> dims, bool autoWindow)
    : FloatTexture3D(dims[0] > 0 ? dims[0] : 1, dims[1] > 0 ? dims[1] : 1, dims[2] > 0 ? dims[2] : 1, 1)
    , samples(samples)
    , ref(this)
    , autoWindow(autoWindow)
{
    SetWrapping(CLAMP);
    SetFiltering(NONE);
    Handle(SamplesChangedEventArg(0, dims[0] * dims[1] * dims[2]));
}

Sample3DTexture::~Sample3DTexture() {
    ref.reset();
}

void Sample3DTexture::Handle(SamplesChangedEventArg arg) {
    logger.info << " samples changed: " << arg.begin << " - " << arg.end << logger.end;
    for (unsigned int i = arg.begin; i < arg.end; ++i)
        ((float*)data)[i] = abs(samples[i]);

    if (autoWindow) {
        unsigned int sliceBegin = arg.begin / (width*height);
        unsigned int sliceEnd = arg.end / (width*height);
        for (unsigned int slice = sliceBegin; slice < sliceEnd; ++slice) {
            logger.info << "slice: " <<  slice << logger.end;
            float min = INFINITY;
            float max = -INFINITY;
            // fetch max and min values
            float* d = (float*)data;
            for (unsigned int i = 0; i < width * height; ++i) {
                    min = fmin(min, d[slice * width * height + i]);
                    max = fmax(max, d[slice * width * height + i]);
            }
            float dist = max - min;
            if (dist > 0) {
                for (unsigned int i = 0; i < width * height; ++i) {
                    d[slice * width * height + i] = (d[slice * width * height + i] - min) / dist;
                }
            }
        }
    }
    this->ChangedEvent().Notify(Texture3DChangedEventArg(ref));
}

} // NS Science
} // NS MRI

