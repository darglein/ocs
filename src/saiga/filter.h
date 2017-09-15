/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image.h"



namespace Saiga {
namespace CUDA {

#define MAX_RADIUS 10
#define MAX_KERNEL_SIZE (MAX_RADIUS*2+1)



 void convolveSinglePassSeparateOuterLinear(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);
 void convolveSinglePassSeparateOuterHalo(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);
 void convolveSinglePassSeparateInner(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);
 void convolveSinglePassSeparateInner75(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);

 void convolveRow(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);
 void convolveCol(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);


 thrust::device_vector<float> createGaussianBlurKernel(int radius, float sigma);

// void setGaussianBlurKernel(float sigma, int radius);

//uploads kernel and convoles images
 void applyFilterSeparate(ImageView<float> src, ImageView<float> dst, ImageView<float> tmp, array_view<float> kernelRow, array_view<float> kernelCol);
 void applyFilterSeparateSinglePass(ImageView<float> src, ImageView<float> dst, array_view<float> kernel);

//only convolves images with previously uploaded kernels
// void gaussianBlur(ImageView<float> src, ImageView<float> dst, int radius);
 void fill(ImageView<float> img, float value);

 void scaleDown2EveryOther(ImageView<float> src, ImageView<float> dst);

 void scaleUp2Linear(ImageView<float> src, ImageView<float> dst);

 // dst = src1 - src2
 void subtract(ImageView<float> src1, ImageView<float> src2, ImageView<float> dst);

 //subtract multiple images at the same time
 //src.n - 1 == dst.n
 //dst[0] = src[0] - src[1]
 //dst[1] = src[1] - src[2]
 //...
 void subtractMulti(ImageArrayView<float> src, ImageArrayView<float> dst);

}
}
