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


#include "cudaSift.h"
#include "saiga/cuda/device_helper.h"

namespace cudasift {

void buildGaussianPyramid(std::vector<SiftImageType>& gaussianPyramid2,
                          int nOctaveLayers, int numOctaves,
                          std::vector<thrust::device_vector<float>>& octaveBlurKernels);

void buildDoGPyramid(std::vector<SiftImageType>& gaussianPyramid2,
                     std::vector<SiftImageType>& dogPyramid2,
                                             int nOctaveLayers, int numOctaves);

void FindPointsMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::ImageArrayView<float> images,
                     unsigned int* pointCounter,
                     float contrastThreshold, float edgeThreshold, int octave, int layers, float sigma, int maxFeatures);


void ComputeOrientationMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::ImageArrayView<float> images,
                             int start, int length,
                             unsigned int* pointCounter,
                             int nOctaveLayers, float sigma, int nfeatures);

void descriptorsMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::array_view<float> descriptors, Saiga::ImageArrayView<float> images,
                      int start, int length);


__global__
void d_scaleDownKeypoints(Saiga::array_view<SiftPoint> keypoints)
{
    int i = threadIdx.x + 128 * blockIdx.x;
    if(i >= keypoints.size())
        return;
    SiftPoint &kpt = keypoints[i];
    float scale = 0.5f;
    kpt.octave = (kpt.octave & ~255) | ((kpt.octave - 1) & 255);
    kpt.xpos *= scale;
    kpt.ypos *= scale;
    kpt.size *= scale;

}

static void scaleDownKeypoints(Saiga::array_view<SiftPoint> keypoints){
    const int BLOCK_SIZE = 128;
    int numblocks = Saiga::iDivUp(keypoints.size(),BLOCK_SIZE);
    d_scaleDownKeypoints<<<numblocks,BLOCK_SIZE>>>(keypoints);
    CUDA_SYNC_CHECK_ERROR();
}


int SIFT_CUDA::compute(SiftImageType d_img, Saiga::array_view<SiftPoint> keypoints, Saiga::array_view<float> descriptors) {
    initMemory();
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("SIFT_CUDA::compute");
#endif
    createInitialImage(d_img,gaussianPyramid2[0],gaussianPyramid2[1]);
    buildGaussianPyramid(gaussianPyramid2,nOctaveLayers,numOctaves,octaveBlurKernels);
    buildDoGPyramid(gaussianPyramid2,dogPyramid2,nOctaveLayers,numOctaves);
    int n = findScaleSpaceExtrema(keypoints,descriptors);
    if( doubleScale ){
        scaleDownKeypoints(keypoints);
    }

    CUDA_SYNC_CHECK_ERROR();
    return n;
}



void SIFT_CUDA::createInitialImage(SiftImageType src, SiftImageType dst, SiftImageType tmp){
#ifdef SIFT_DEBUG
    cout << "createInitialImage. lowimg: " << dst.cols << "x" << dst.rows << " img: " << src.cols << "x" << src.rows << " sigma: " << sigma << endl;
#endif

#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("SIFT_CUDA::createInitialImage");
#endif

    if (!doubleScale) {
        Saiga::CUDA::applyFilterSeparateSinglePass(src,dst,initialBlurKernel);
    }else{
        //note: the blur takes up roughly 2x the time of scale up
        Saiga::CUDA::scaleUp2Linear(src,tmp);
        Saiga::CUDA::applyFilterSeparateSinglePass(tmp,dst,initialBlurKernel);
    }

#ifdef SIFT_DEBUG
    {
        cv::Mat cpumat = Saiga::ImageViewToMat(dst);
        Saiga::CUDA::copyImage(dst,Saiga::MatToSiftImageType(cpumat),cudaMemcpyDeviceToHost);
        cv::imwrite("out/init_sift_img_blurred_gpu.jpg",cpumat);
    }
#endif
    CUDA_SYNC_CHECK_ERROR();

}


int SIFT_CUDA::findScaleSpaceExtrema(Saiga::array_view<SiftPoint> keypoints, Saiga::array_view<float> descriptors)
{
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("SIFT_CUDA extrema detection + descriptors");
#endif


    thrust::fill(pointCounter.begin(),pointCounter.end(),0);
    CUDA_SYNC_CHECK_ERROR();
    int n = 0;

    for(int o = 0; o < numOctaves;++o){
        n = pointCounter[0];
        int pointsBefore = n;

#ifdef SIFT_DEBUG
        cout << "Extracting Points of octave " << o << ". Points before: " << pointsBefore << endl;
#endif

        auto dst2 = Saiga::ImageArrayView<float>(dogPyramid2[o*(nOctaveLayers + 2)], nOctaveLayers + 2);
        FindPointsMulti(keypoints,dst2,thrust::raw_pointer_cast(pointCounter.data()),contrastThreshold,edgeThreshold,o,nOctaveLayers,sigma,nfeatures);

        n = pointCounter[0];
        int newPoints = n - pointsBefore;
        SAIGA_ASSERT(newPoints >= 0);

#ifdef SIFT_DEBUG
        cout << "Found " << newPoints << " new points." << endl;
#endif
        if(newPoints > 0){
            auto img2 = Saiga::ImageArrayView<float>(gaussianPyramid2[o*(nOctaveLayers + 3)], nOctaveLayers + 3);
            ComputeOrientationMulti(keypoints,img2,pointsBefore,newPoints,thrust::raw_pointer_cast(pointCounter.data()),nOctaveLayers,sigma,nfeatures);
            n = pointCounter[0];
            newPoints = n - pointsBefore;
            descriptorsMulti(keypoints,descriptors,img2,pointsBefore,newPoints);
        }
    }

    n = pointCounter[0];
    CUDA_SYNC_CHECK_ERROR();
    return n;
}

}
