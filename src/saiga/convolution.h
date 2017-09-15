/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cudaHelper.h"
#include "saiga/imageView.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>


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

}
}
