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
#include "saiga/cuda/reduce.h"





namespace cudasift {
template<unsigned int THREADS_PER_BLOCK, int MAX_RADIUS>
__device__ inline
void calcSIFTDescriptorBlock( SiftImageType d_img,
                              int px, int py, float ori, float scl, float* hist, float* dst, float* sreduce, int local_thread_id, int lane_id ){

    const int d = SIFT_DESCR_WIDTH; //4
    const int n = SIFT_DESCR_HIST_BINS; //8

    const float exp_scale = -1.f/(d * d * 0.5f);
    const float bins_per_rad = n / 360.f;
    const int histlen = (d+2)*(d+2)*(n+2);

    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = Saiga::iRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    if(radius > MAX_RADIUS){
        radius = MAX_RADIUS;
        hist_width = MAX_RADIUS / (1.4142135623730951f * (d + 1) * 0.5f);
    }

    float cos_t = cosf(ori*(float)(SIFT_PI / 180));
    float sin_t = sinf(ori*(float)(SIFT_PI / 180));
    cos_t /= hist_width;
    sin_t /= hist_width;

    // Clip the radius to the diagonal of the image
    radius = min(radius, (int) sqrtf(d_img.cols*d_img.cols + d_img.rows*d_img.rows));

    int numSample = (radius*2+1)*(radius*2+1);


    //    for(int l = lane_id; l < histlen; l+=WARP_SIZE )
    //    WARP_FOR(l,lane_id,histlen,WARP_SIZE)
    WARP_FOR(l,local_thread_id,histlen,THREADS_PER_BLOCK)
    {
        int k = l % (n+2);
        int j = (l / (n+2)) % (d+2);
        int i = l / ((n+2)*(d+2));
        hist[(i*(d+2) + j)*(n+2) + k] = 0.f;
    }

    if(local_thread_id < 2){
        sreduce[local_thread_id] = 0;
    }

    __syncthreads();


    //    for(int  k = local_thread_id; k < numSample; k+=THREADS_PER_BLOCK)
    WARP_FOR(k,local_thread_id,numSample,THREADS_PER_BLOCK)
    {
        int i = k / (radius*2+1) - radius;
        int j = k % (radius*2+1) - radius;
        // Calculate sample's histogram array coords rotated relative to ori.
        // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
        // r_rot = 1.5) have full weight placed in row 1 after interpolation.
        float c_rot = j * cos_t - i * sin_t;
        float r_rot = j * sin_t + i * cos_t;
        float rbin = r_rot + d/2 - 0.5f;
        float cbin = c_rot + d/2 - 0.5f;
        int x = px + j;
        int y = py + i;

        if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                y > 0 && y < d_img.rows - 1 && x > 0 && x < d_img.cols - 1 )
        {
            float dx = d_img(y,x+1) - d_img(y,x-1);
            float dy = d_img(y-1,x) - d_img(y+1,x);

            float w = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
            w = exp(w);

            float Ori = atan2f(dy, dx)/(2.0f * 3.1416f) * 360.0f;
            Ori = Ori < 0 ? Ori + 360.0f : Ori; //convert [-180,180] to [0,360]
            float Mag = sqrtf(dx*dx + dy*dy);


            float obin = (Ori - ori)*bins_per_rad;
            float mag = Mag*w;

            int r0 = Saiga::iFloor( rbin );
            int c0 = Saiga::iFloor( cbin );
            int o0 = Saiga::iFloor( obin );

            rbin -= r0;
            cbin -= c0;
            obin -= o0;

            if( o0 < 0 )
                o0 += n;
            if( o0 >= n )
                o0 -= n;


            // histogram update using tri-linear interpolation
            float v_r1 = mag*rbin, v_r0 = mag - v_r1;
            float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
            float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
            float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
            float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
            float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
            float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

            int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;

            //this is currently the main bottleneck
            atomicAdd(&hist[idx] , v_rco000);
            atomicAdd(&hist[idx+1] , v_rco001);
            atomicAdd(&hist[idx+(n+2)] , v_rco010);
            atomicAdd(&hist[idx+(n+3)] , v_rco011);

            atomicAdd(&hist[idx+(d+2)*(n+2)] , v_rco100);
            atomicAdd(&hist[idx+(d+2)*(n+2)+1] , v_rco101);
            atomicAdd(&hist[idx+(d+3)*(n+2)] , v_rco110);
            atomicAdd(&hist[idx+(d+3)*(n+2)+1] , v_rco111);
        }
    }


    __syncthreads();


    // finalize histogram, since the orientation histograms are circular
    WARP_FOR(l,local_thread_id,d*d,THREADS_PER_BLOCK)
    {
        int i = l / d;
        int j = l % d;
        int idx = ((i+1)*(d+2) + (j+1))*(n+2);
        hist[idx] += hist[idx+n];
        hist[idx+1] += hist[idx+n+1];
    }



    __syncthreads();

    // copy histogram to the descriptor,
    {
        int l = local_thread_id;
        int k = l % n;
        int j = (l / (n)) % (d);
        int i = l / ((n)*(d));
        int idx = ((i+1)*(d+2) + (j+1))*(n+2);
        dst[l] = hist[idx+k];
    }


    // apply hysteresis thresholding
    float nrm2 = dst[local_thread_id];
    nrm2 = nrm2 * nrm2;
    {
        //atomic shared memory reduction
        nrm2 = Saiga::CUDA::warpReduceSum<float,WARP_SIZE,false>(nrm2);
        if(lane_id == 0){
            atomicAdd(&sreduce[0],nrm2);
        }
        __syncthreads();
        nrm2 = sreduce[0];
    }


    float thr = sqrtf(nrm2)*SIFT_DESCR_MAG_THR;


    {
        float val = min(dst[local_thread_id], thr);
        dst[local_thread_id] = val;
        nrm2 = val*val;
    }

    {
        //atomic shared memory reduction
        nrm2 = Saiga::CUDA::warpReduceSum<float,WARP_SIZE,false>(nrm2);
        if(lane_id == 0){
            atomicAdd(&sreduce[1],nrm2);
        }
        __syncthreads();
        nrm2 = sreduce[1];
    }



    nrm2 = SIFT_INT_DESCR_FCTR / max(sqrtf(nrm2), SIFT_FLT_EPSILON);

    {
        dst[local_thread_id] = dst[local_thread_id]*nrm2;
    }


}

template<unsigned int THREADS_PER_BLOCK, int MAX_RADIUS>
__global__ void calcSIFTDescriptorsBlock(
        Saiga::ImageArrayView<float> images,
        Saiga::array_view<SiftPoint> d_Sift,
        Saiga::array_view<float> d_desc,
        int pointsBefore, int numPoints
        )
{

    //    Saiga::CUDA::ThreadInfo<THREADS_PER_BLOCK> ti;

    const int d = SIFT_DESCR_WIDTH; //4
    const int n = SIFT_DESCR_HIST_BINS; //8
    const int desclen = (d)*(d)*(n); //128 = 512 byte
    const int histlen = (d+2)*(d+2)*(n+2); //360 = 1440 byte

    //shared memory per warp = 1952
    //warps per sm: 64
    //max active warps: 25
    __shared__ float sdesc[desclen];
    __shared__ float shist[histlen];
    __shared__ float sreduce[THREADS_PER_BLOCK / WARP_SIZE];

    float* desc = sdesc;
    float* hist = shist;


    int id = blockIdx.x;

    if(id >= numPoints)
        return;



    SiftPoint& sp = d_Sift[id+pointsBefore];
    float* rdesc = d_desc.data() + (id+pointsBefore) * SIFT_DESCRIPTOR_SIZE;




    float angle = 360.0f - sp.orientation;
    if (fabsf(angle - 360.f) < SIFT_FLT_EPSILON)
        angle = 0.f;


    int octave, layer;
    float scale;
    sp.unpackOctave(octave, layer, scale);
    float size = sp.size*scale;


    SiftImageType d_img = images[layer];
    calcSIFTDescriptorBlock<THREADS_PER_BLOCK,MAX_RADIUS>(d_img,sp.ixpos,sp.iypos,angle, size*0.5f, hist, desc,sreduce,threadIdx.x,threadIdx.x & (WARP_SIZE-1));

    rdesc[threadIdx.x] = desc[threadIdx.x];
}

void descriptorsMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::array_view<float> descriptors, Saiga::ImageArrayView<float> images, int start, int length){
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("   SIFT_CUDA::descriptorsMulti");
#endif
    const int BLOCK_SIZE = 128;
    calcSIFTDescriptorsBlock<BLOCK_SIZE,SIFT_DESCR_MAX_RADIUS><<<Saiga::iDivUp(length*BLOCK_SIZE, BLOCK_SIZE),BLOCK_SIZE>>>(images,
                                                                                                                            keypoints,
                                                                                                                            descriptors,
                                                                                                                            start,length
                                                                                                                            );
    CUDA_SYNC_CHECK_ERROR();
}

}
