/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/imageView.h"

namespace Saiga {
namespace CUDA {

void fill(ImageView<float> img, float value);

void scaleDown2EveryOther(ImageView<float> src, ImageView<float> dst);

void scaleUp2Linear(ImageView<float> src, ImageView<float> dst);


}
}
