/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "cudaSift.h"

namespace cudasift {

class MatchGPU{
public:
    MatchGPU(int nfeatures);
    void initMemory() ;


    /**
     * K Nearest neighbour search between descriptor array 1 and 2.
     * All arrays must be on the device.
     * out_distance.size >= descriptors1.size() / 128 * k
     */
    void knnMatch(Saiga::array_view<float> descriptors1,Saiga::array_view<float> descriptors2, Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index, int k);



    /**
     * Match the given keypoints and descriptors.
     * Maximum distance between kp1 and kp2 (not the distance between their descriptors!!!) is given by r.
     */
    void radiusMatch(Saiga::array_view<SiftPoint> keypoints1, Saiga::array_view<float> descriptors1, Saiga::array_view<SiftPoint> keypoints2, Saiga::array_view<float> descriptors2, Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index, int k, float r);

private:
//    thrust::device_vector<float> distancesMatrix;
    Saiga::CUDA::CudaImage<float> distances;
    int maxPoints;

    void computeDistanceMatrix(Saiga::array_view<float> descriptors1, Saiga::array_view<float> descriptors2);
    void computeKNN(Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index, int k);
    void filterByRadius(Saiga::array_view<SiftPoint> keypoints1, Saiga::array_view<SiftPoint> keypoints2, float r);
};

}
