/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
 Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

#pragma once

#include "saiga/common.h"
#include "saiga/image.h"

#include "sift_defines.h"

#ifdef SIFT_DEBUG
#include "saiga/opencv/opencv.h"
#endif

using std::cout;
using std::endl;

namespace cudasift {




using Saiga::ImageView;


//size = 8 * sizeof(int) = 32 bytes
struct __attribute__((aligned(32))) SiftPoint {

    //output position with subpixel accuracy
    float xpos;
    float ypos;
    //local pixel position in the current octave
    int ixpos;
    int iypos;

    //see cv::Keypoint for more details
    int octave;
    float size;
    float orientation;
    float response;

    HD inline
    void unpackOctave(int& octave, int& layer, float& scale)
    {
        octave = this->octave & 255;
        layer = (this->octave >> 8) & 255;
        octave = octave < 128 ? octave : (-128 | octave);
        scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
    }

    HD inline
    void packOctave(int octave, int layer){
        this->octave = octave + (layer << 8);
    }
};



class SIFTGPU{
public:
    SIFTGPU(
            int imageWidth, int imageHeight, bool doubleScale, int maxOctaves,
            int nfeatures = 0, int nOctaveLayers = 3,
            double contrastThreshold = 0.04, double edgeThreshold = 10,
            double sigma = 1.6);

    ~SIFTGPU();

    void initMemory();
    int compute(ImageView<float> img, Saiga::array_view<SiftPoint> keypoints, Saiga::array_view<float> descriptors);
private:
    void createKernels();
    void createInitialImage(ImageView<float> src, ImageView<float> dst, ImageView<float> tmp);
    void buildGaussianPyramid();
    void buildDoGPyramid();
    int findScaleSpaceExtrema(Saiga::array_view<SiftPoint> keypoints, Saiga::array_view<float> descriptors);
    void FindPointsMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::ImageArrayView<float> images, int o);
    void ComputeOrientationMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::ImageArrayView<float> images, int start, int length);
    void descriptorsMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::array_view<float> descriptors, Saiga::ImageArrayView<float> images, int start, int length);


    std::vector<ImageView<float>> gaussianPyramid2;
    std::vector<ImageView<float>> dogPyramid2;


    thrust::device_vector<unsigned int> pointCounter;
    thrust::device_vector<float> initialBlurKernel;
    std::vector<thrust::device_vector<float>> octaveBlurKernels;

    thrust::device_vector<uint8_t> memorygpyramid;
    thrust::device_vector<uint8_t> memorydogpyramid;

#ifndef SIFT_SINGLE_PASS_BLUR
    thrust::device_vector<uint8_t> memoryTmp; //for gaussian blur
    std::vector<ImageView<float>> tmpImages;
#endif


    int numOctaves;
    int imageWidth;
    int imageHeight;
    bool doubleScale;
    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;

    bool initialized = false;
};

}
