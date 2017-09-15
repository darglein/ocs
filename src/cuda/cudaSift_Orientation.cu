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
#include "saiga/device_helper.h"



namespace cudasift {


// Computes a gradient orientation histogram at a specified pixel
__device__
static float calcOrientationHistWarp(  ImageView<float> d_img,
                                       int px, int py, int RADIUS,
                                       float sigma, float* hist, float* temphist,
                                       unsigned int lane_id )
{
    //    radius = 8;
    const int n = SIFT_ORI_HIST_BINS;
    //    const int RADIUS = 8;
    const int len = (RADIUS*2+1)*(RADIUS*2+1);
    int i, j;

    float expf_scale = -1.f/(2.f * sigma * sigma);

//    for( i = lane_id; i < n+4; i+=WARP_SIZE ){
    WARP_FOR(i,lane_id,n+4,WARP_SIZE)
    {
        temphist[i] = 0.f;
    }

//    for(  k = lane_id; k < len; k+=WARP_SIZE )
    WARP_FOR(k,lane_id,len,WARP_SIZE)
    {
        i = k / (RADIUS*2+1) - RADIUS;
        j = k % (RADIUS*2+1) - RADIUS;

        int y = py + i;
        int x = px + j;

        if( x <= 0 || x >= d_img.width - 1 )
            continue;

        if( y <= 0 || y >= d_img.height - 1 )
            continue;

        float dx = d_img.atIVxxx(y,x+1) - d_img.atIVxxx(y,x-1);
        float dy = d_img.atIVxxx(y-1,x) - d_img.atIVxxx(y+1,x);

        float w = (i*i + j*j)*expf_scale;
        w = expf(w);

        float Ori = atan2f(dy, dx)/(2.0f * 3.1416f) * 360.0f;
        Ori = Ori < 0 ? Ori + 360.0f : Ori; //convert [-180,180] to [0,360]
        float Mag = sqrtf(dx*dx + dy*dy);

        int bin = Saiga::iRound((n/360.f)*Ori);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;

        atomicAdd(&temphist[bin+2],w * Mag);
    }


    // smooth the histogram
    if(lane_id < 2){
        temphist[lane_id & 1] = temphist[n + (lane_id & 1)];
        temphist[n + 2 + (lane_id & 1)] = temphist[2 + (lane_id & 1)];
    }


//        for( i = lane_id; i < n; i+=WARP_SIZE )
    WARP_FOR(i,lane_id,n,WARP_SIZE)
    {
        int j = i + 2;
        hist[i] = (temphist[j-2] + temphist[j+2])*(1.f/16.f) +
                (temphist[j-1] + temphist[j+1])*(4.f/16.f) +
                temphist[j]*(6.f/16.f);
    }


    float maxval = hist[lane_id];

//    for( i = lane_id + WARP_SIZE; i < n; i+=WARP_SIZE )
    WARP_FOR(i,lane_id + WARP_SIZE,n,WARP_SIZE)
    {
        maxval = max(maxval,hist[i]);
    }

    maxval = Saiga::CUDA::warpReduceMax<float,WARP_SIZE,true>(maxval);

    return maxval;
}



template<unsigned int THREADS_PER_BLOCK, int MAX_RADIUS>
__global__ void ComputeOrientationWarp(
        Saiga::ImageArrayView<float> images,
        Saiga::array_view<SiftPoint> d_Sift,
        unsigned int* pointCounter,
        int pointsBefore, int numPoints, int nOctaveLayers,
        float sig, int maxFeatures
        )
{

//    Saiga::CUDA::ThreadInfo<THREADS_PER_BLOCK> ti;

    int local_thread_id = threadIdx.x;
    int   lane_id         = local_thread_id & (WARP_SIZE-1);
    int thread_id       = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    int warp_id         = thread_id   / WARP_SIZE;
    int warp_lane       = threadIdx.x / WARP_SIZE;

    int id = warp_id;

    if(id >= numPoints)
        return;


    const int n = SIFT_ORI_HIST_BINS;

    __shared__ float shist[THREADS_PER_BLOCK / WARP_SIZE * n];
    __shared__ float stemphist[THREADS_PER_BLOCK / WARP_SIZE * (n+4)];
    __shared__ int outPointsa[THREADS_PER_BLOCK / WARP_SIZE];

    float* hist = shist + warp_lane * n;
    float* temphist = stemphist + warp_lane * (n+4);

    SiftPoint& sp = d_Sift[id+pointsBefore];

    int layer, octave;
    float scale;
    sp.unpackOctave(octave,layer,scale);

    ImageView<float> d_img = images[layer];


    //variable radius
    float size = sig*powf(2.f, float(layer) / nOctaveLayers)*(1 << octave)*2;
    float scl_octv = size*0.5f/(1 << octave);
    float sigma = SIFT_ORI_SIG_FCTR * scl_octv;
    int radius = Saiga::iRound(SIFT_ORI_RADIUS * scl_octv);
    if(radius > MAX_RADIUS){
        radius = MAX_RADIUS;
        scl_octv = MAX_RADIUS / SIFT_ORI_RADIUS;
        sigma = SIFT_ORI_SIG_FCTR * scl_octv;
    }
    float omax = calcOrientationHistWarp(d_img,sp.ixpos,sp.iypos,radius,sigma,
                                         hist,temphist,lane_id);

    float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);

    int& outPoints = outPointsa[warp_lane];
    if(lane_id == 0)
        outPoints = 0;



//    for( int j = ti.lane_id; j < n; j+=WARP_SIZE )
    WARP_FOR(j,lane_id,n,WARP_SIZE)
    {
        int leftBin = j > 0 ? j - 1 : n - 1;
        int rightBin = j < n-1 ? j + 1 : 0;

        if( hist[j] > hist[leftBin]  &&  hist[j] > hist[rightBin]  &&  hist[j] >= mag_thr )
        {
            float bin = j + 0.5f * (hist[leftBin]-hist[rightBin]) / (hist[leftBin] - 2*hist[j] + hist[rightBin]);
            bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
            float angle = 360.f - (float)((360.f/n) * bin);
            if(fabsf(angle - 360.f) < __FLT_EPSILON__)
                angle = 0.f;

            CUDA_ASSERT(angle >= 0 && angle <= 360);

            int prevPoints = atomicAdd(&outPoints,1);

            if(prevPoints == 0){
                sp.orientation = angle;
            }else{
                int idx = atomicInc(pointCounter, 0x7fffffff);
                if(idx < maxFeatures){
                    SiftPoint& newPoint = d_Sift[idx];
                    newPoint = sp;
                    newPoint.orientation = angle;
                }else{
                    atomicDec(pointCounter, 0x7fffffff);
                }
            }
        }
    }

    if (lane_id == 0 && outPoints == 0)
	{
		sp.orientation = 0;
	}

#ifdef CUDA_DEBUG
	//__syncthreads();
    //CUDA_ASSERT(outPoints > 0);
#endif
}




void SIFTGPU::ComputeOrientationMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::ImageArrayView<float> images, int start, int length){
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("  SIFTGPU::ComputeOrientationMulti");
#endif
    const int BLOCK_SIZE = 128;
	//cout << start << " " << length << endl;
    ComputeOrientationWarp<BLOCK_SIZE,SIFT_ORI_MAX_RADIUS><<<Saiga::iDivUp(length*WARP_SIZE, BLOCK_SIZE),BLOCK_SIZE>>>(images,
                                                                                                                       keypoints,
                                                                                                                       thrust::raw_pointer_cast(pointCounter.data()),
                                                                                                                       start,length,nOctaveLayers,
                                                                                                                       sigma,nfeatures);
    CUDA_SYNC_CHECK_ERROR();
	
}

}
