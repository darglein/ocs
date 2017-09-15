/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/imageView.h"


namespace Saiga {
namespace CUDA {


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
