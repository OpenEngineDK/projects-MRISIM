// MRI visualize sampler data
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_SAMPLER_VISUALIZER_H_
#define _MRI_SAMPLER_VISUALIZER_H_

#include "MRISim.h"
#include "../Resources/Sample3DTexture.h"

namespace MRI {
namespace Science {

    using namespace Resources;

class SamplerVisualizer: public IListener<SamplesChangedEventArg> {
private:
    IMRISampler& sampler;
    Sample3DTexturePtr sampleTex, imageTex;
public:
    SamplerVisualizer(IMRISampler& sampler)
        : sampler(sampler)
        , sampleTex(Sample3DTexturePtr(new Sample3DTexture(sampler.GetSamples(), sampler.GetDimensions(), true)))
        , imageTex(Sample3DTexturePtr(new Sample3DTexture(sampler.GetReconstructedSamples(), sampler.GetDimensions(), true)))
    {
        sampler.SamplesChangedEvent().Attach(*this);
    }
    
    virtual ~SamplerVisualizer() {}
    
    Sample3DTexturePtr GetSamplesTexture() { return sampleTex; }
    Sample3DTexturePtr GetImageTexture() { return imageTex; }

    void Handle(SamplesChangedEventArg arg) {
        sampleTex->Handle(arg);
        imageTex->Handle(SamplesChangedEventArg(sampler.GetReconstructedSamples(), arg.dims, 0, arg.dims[0] * arg.dims[1] * arg.dims[2]));
    }
};

} // NS Science
} // NS MRI

#endif // _MRI_CARTESIAN_SAMPLER_H_
