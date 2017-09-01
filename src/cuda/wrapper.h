/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "opencv2/opencv.hpp"
#include "cuda/cudaSift.h"
#include "saiga/cuda/imageProcessing/image.h"


class SIFTGPU;

namespace SiftWrapper{

void KeypointsToCV(Saiga::array_view<SiftPoint> keypoints, std::vector<cv::KeyPoint> &cvkeypoints);

}
