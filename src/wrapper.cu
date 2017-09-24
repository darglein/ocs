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




namespace cudasift {

void SIFT_CUDA::downloadKeypoints(Saiga::array_view<SiftPoint> keypointsGPU, std::vector<cv::KeyPoint>& keypoints){
    int numPoints = keypointsGPU.size();

    std::vector<SiftPoint> hkeypoints(numPoints);
    thrust::copy(keypointsGPU.tbegin(),keypointsGPU.tbegin()+numPoints,hkeypoints.begin());


    keypoints.resize(numPoints);
    for(int i = 0; i < numPoints;++i){
        cv::KeyPoint kp;
        SiftPoint& sp = hkeypoints[i];
        kp.pt = cv::Point2f(sp.xpos,sp.ypos);
        kp.size = sp.size;
        kp.angle = sp.orientation;
        kp.octave = sp.octave;
        kp.response = sp.response;
        keypoints[i] = kp;
    }
}


}
