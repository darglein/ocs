/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/imageView.h"
#include "saiga/cudaHelper.h"


namespace Saiga {
namespace CUDA {


 thrust::device_vector<float> createGaussianBlurKernel(int radius, float sigma);

// void setGaussianBlurKernel(float sigma, int radius);

//uploads kernel and convoles images
 void applyFilterSeparate(ImageView<float> src, ImageView<float> dst, ImageView<float> tmp, array_view<float> kernelRow, array_view<float> kernelCol);
 void applyFilterSeparateSinglePass(ImageView<float> src, ImageView<float> dst, array_view<float> kernel);

//only convolves images with previously uploaded kernels
// void gaussianBlur(ImageView<float> src, ImageView<float> dst, int radius);

}
}
