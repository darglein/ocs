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
#include "saiga/cuda/imageProcessing/imageProcessing.h"


using std::cout;
using std::endl;

SIFTGPU::SIFTGPU(int imageWidth, int imageHeight, bool doubleScale, int maxOctaves,
                 int _nfeatures, int _nOctaveLayers,
                 double _contrastThreshold, double _edgeThreshold, double _sigma )
    : SIFTBase(imageWidth,imageHeight,doubleScale,_nfeatures,_nOctaveLayers,_contrastThreshold,_edgeThreshold,_sigma)
{
    numOctaves =  Saiga::iRound(std::log( (double)std::min( imageWidth, imageHeight ) ) / std::log(2.) - 2) + 1;
    if(maxOctaves > 0)
        numOctaves = std::min(numOctaves,maxOctaves);
}

SIFTGPU::~SIFTGPU(){

    CHECK_CUDA_ERROR(cudaFree(memorydogpyramid));
    CHECK_CUDA_ERROR(cudaFree(memorygpyramid));
    memorydogpyramid = 0;
    memorygpyramid = 0;
}

void SIFTGPU::initMemory()
{
    if(initialized)
        return;
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("SIFTGPU::initMemory");
#endif



#ifdef SIFT_DEBUG
    std::cout << " ==== ExtractSift nOctaves=" << numOctaves << " octave layers: " << nOctaveLayers << std::endl;
#endif
    int width = imageWidth*(doubleScale ? 2 : 1);
    int height = imageHeight*(doubleScale ? 2 : 1);

    //size of the gaussian pyramid in float
    size_t pyramidSize = 0;

    for (int o=0, w = width, h = height; o<numOctaves; o++) {
        int pw = Saiga::iAlignUp(w, 128);
        size_t imageSize = h*pw;
        pyramidSize += (nOctaveLayers + 3) * imageSize;
#ifdef SIFT_DEBUG
        cout << "Octave " << o << " - ImageSize: " << w << "x" << h << ", PaddedImageSize: " <<  pw << "x" << h  << ", MemoryPerImage: " << imageSize << ", MemoryPerOctave: " << imageSize*(nOctaveLayers + 3)  << endl;
#endif
        w /= 2;
        h /= 2;
    }

    size_t pyramidSizeBytes = pyramidSize * sizeof(float);

#ifdef SIFT_DEBUG
    cout << "Memory for gaussian pyramid: " << pyramidSizeBytes << " ~ " << double(pyramidSizeBytes) / 1000.0 / 1000.0 << "mb" << endl;
#endif

    CHECK_CUDA_ERROR(cudaMalloc((void **)&memorydogpyramid, pyramidSizeBytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&memorygpyramid, pyramidSizeBytes));


    gaussianPyramid2.resize(numOctaves * (nOctaveLayers + 3));
    dogPyramid2.resize(numOctaves * (nOctaveLayers + 2));

    pointCounter.resize(1);

    size_t ps = 0;
    size_t dps = 0;
    for (int o=0, w = width, h = height; o<numOctaves; o++) {

        int pw = Saiga::iAlignUp(w, 128);
        size_t imageSize = h*pw;

        for(int j = 0; j < nOctaveLayers + 3 ; ++j){
            int index = o * (nOctaveLayers + 3) + j;
            gaussianPyramid2[index] = ImageView<float>(w,h,pw*sizeof(float),memorygpyramid+ps);
            ps += imageSize;
        }

        for(int j = 0; j < nOctaveLayers + 2 ; ++j){
            int index = o * (nOctaveLayers + 2) + j;
            dogPyramid2[index] = ImageView<float>(w,h,pw*sizeof(float),memorydogpyramid+dps);
            dps += imageSize;
        }

        w /= 2;
        h /= 2;
    }


    createKernels();

    initialized = true;
    CUDA_SYNC_CHECK_ERROR();
}

void SIFTGPU::createKernels(){
    if (!doubleScale) {
        float sig_diff = sqrtf( std::max<double>(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        initialBlurKernel = Saiga::CUDA::createGaussianBlurKernel(GAUSSIAN_KERNEL_RADIUS,sig_diff);
    }else{
        float sig_diff = sqrtf( std::max<double>(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        initialBlurKernel = Saiga::CUDA::createGaussianBlurKernel(GAUSSIAN_KERNEL_RADIUS,sig_diff);
    }


    std::vector<double> sig(nOctaveLayers + 3);
    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow( 2., 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }
    octaveBlurKernels.resize(nOctaveLayers + 3);

    for( int i = 0; i < nOctaveLayers + 3; i++ )
    {
        octaveBlurKernels[i] = Saiga::CUDA::createGaussianBlurKernel(GAUSSIAN_KERNEL_RADIUS,sig[i]);
    }
}


