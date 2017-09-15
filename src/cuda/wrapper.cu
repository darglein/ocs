/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "cudaSift.h"
#include "matching.h"

#include "saiga/opencv.h"
#include "wrapper.h"


namespace cudasift {
namespace SiftWrapper{

void KeypointsToCV(Saiga::array_view<SiftPoint> keypoints, std::vector<cv::KeyPoint>& cvkeypoints){
    cvkeypoints.resize(keypoints.size());
    for(int i = 0; i < keypoints.size();++i){
        cv::KeyPoint kp;
        SiftPoint& sp = keypoints[i];
        kp.pt = cv::Point2f(sp.xpos,sp.ypos);
        kp.size = sp.size;
        kp.angle = sp.orientation;
        kp.octave = sp.octave;
        kp.response = sp.response;
        cvkeypoints[i] = kp;
    }
}

}
}
