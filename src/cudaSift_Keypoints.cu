﻿/**
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
#include "saiga/cuda/memory.h"
#include "saiga/cuda/math/inverse.h"

namespace cudasift {

HD inline
void solveSymmetric(
        float a, float b, float c, float d, float e, float f,
        float h, float j, float k,
        float& x, float& y, float& z
        )
{
    float ad = a*d;
    float ae = a*e;
    float af = a*f;

    float bc = b*c;
    float be = b*e;
    float bf = b*f;
    float bb = b*b;

    float cd = c*d;
    float ce = c*e;
    float cc = c*c;

    float df = d*f;
    float ee = e*e;

    float div = -ad*f + ae*e + b*bf - 2 * bc*e + c*cd;
    div = 1.0f / div;

    float becd = (-be + cd);
    float bfce = (bf - ce);
    float aebc = (ae - bc);

    x = k*becd + j*bfce + h*(-df + ee);
    y = k*aebc + j*(-af + cc) + h*bfce;
    z = k*(-ad + bb) + j*aebc + h*becd;

    x *= div;
    y *= div;
    z *= div;
}

template<unsigned int TILE_W, unsigned int TILE_H, unsigned int LAYERS>
__device__ inline
void loadBufferShared(float sbuffer[LAYERS + 2][TILE_H][TILE_W], float buffer[3][3][3], int x, int y, int layer)
{
#pragma unroll
    for(int i = 0; i < 3 * 3 * 3; ++i){
        int dx = i % 3 - 1;
        int dy = i / 3 % 3 - 1;
        int dz = i / 3 / 3 % 3 - 1;
        float v = sbuffer[layer+dz][y+dy][x+dx];
        buffer[dx+1][dy+1][dz+1] = v;
    }
}

template<unsigned int TILE_W, unsigned int TILE_H, unsigned int LAYERS>
__device__ inline
void loadBufferSharedMixed(
        Saiga::ImageArrayView<float>& images,
        float sbuffer[LAYERS + 2][TILE_H][TILE_W],
float buffer[3][3][3], int x, int y, int layer,
int x_tile, int y_tile)
{
#pragma unroll
    for(int i = 0; i < 3 * 3 * 3; ++i){
        int dx = i % 3 - 1;
        int dy = i / 3 % 3 - 1;
        int dz = i / 3 / 3 % 3 - 1;

        int lx = x + dx;
        int ly = y + dy;
        int lz = layer + dz;

        float v = (lx >= 0 && lx < TILE_W && ly >= 0 && ly < TILE_H) ?
                    sbuffer[lz][ly][lx] :
                    images[lz]( ly + y_tile,lx + x_tile);
        buffer[dx+1][dy+1][dz+1] = v;
    }
}

__device__ inline
void findMinMax(float buffer[3][3][3], float& minN, float& maxN)
{
    minN = 1253453453;
    maxN = -1545523;
#pragma unroll
    for(int i = 0; i < 3 * 3 * 3; ++i){
        int dx = i % 3 ;
        int dy = i / 3 % 3 ;
        int dz = i / 3 / 3 % 3 ;
        if(dx == 1 && dy == 1 && dz == 1){
        }else{
            auto v = buffer[dx][dy][dz];
            minN = min(minN,v);
            maxN = max(maxN,v);
        }
    }
}

__device__ inline
void derive(float buffer[3][3][3], float3& dD)
{
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;

    dD.x = (buffer[2][1][1] - buffer[0][1][1]) * deriv_scale;
    dD.y = (buffer[1][2][1] - buffer[1][0][1]) * deriv_scale;
    dD.z = (buffer[1][1][2] - buffer[1][1][0]) * deriv_scale;
}

__device__ inline
void deriveSecond(float buffer[3][3][3], float derivs2[6])
{
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;
    float v2 = buffer[1][1][1] * 2.0f;

    derivs2[0] = (buffer[2][1][1] + buffer[0][1][1] - v2)*second_deriv_scale;
    derivs2[1] = (buffer[1][2][1] + buffer[1][0][1] - v2)*second_deriv_scale;
    derivs2[2] = (buffer[1][1][2] + buffer[1][1][0] - v2)*second_deriv_scale;

    derivs2[3] = (buffer[2][2][1] - buffer[0][2][1] -
            buffer[2][0][1] + buffer[0][0][1] ) * cross_deriv_scale;
    derivs2[4] = (buffer[2][1][2] - buffer[0][1][2] -
            buffer[2][1][0] + buffer[0][1][0] ) * cross_deriv_scale;
    derivs2[5] = (buffer[1][2][2] - buffer[1][0][2] -
            buffer[1][2][0] + buffer[1][0][0] ) * cross_deriv_scale;
}

template<unsigned int TILE_W, unsigned int TILE_H, unsigned int LAYERS>
__global__ static
__launch_bounds__(TILE_W*TILE_H,3) //make sure nvcc reduces register usage to atleast 75% occupancy
void d_FindPointsMulti4(
        Saiga::ImageArrayView<float> images,
        Saiga::array_view<SiftPoint> keypoints,
        unsigned int* pointCounter,
        float contrastThreshold, float edgeThreshold, int octave, float sigma, int maxFeatures, int threshold)
{
    float buffer[1][3][3][3];
    const int bufferOffset = 0;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const int RADIUS = 1;
    int x_tile = blockIdx.x * (TILE_W - 2 * RADIUS) - RADIUS + SIFT_IMG_BORDER;
    int y_tile = blockIdx.y * (TILE_H - 2 * RADIUS) - RADIUS + SIFT_IMG_BORDER;

    int xo = x_tile + tx;
    int yo = y_tile + ty;

    __shared__ float sbuffer[LAYERS + 2][TILE_H][TILE_W];
    for(int i = 0; i < LAYERS + 2; ++i){
        sbuffer[i][ty][tx]  = images[i].clampedRead(yo,xo);
    }

    //only process inner pixels
    if(ty < RADIUS || ty >= TILE_H - RADIUS)
        return;
    if(tx < RADIUS || tx >= TILE_W - RADIUS)
        return;

    __syncthreads();

    //don't produce keypoint at the border
    if(xo < SIFT_IMG_BORDER
            || yo < SIFT_IMG_BORDER
            || xo >= images.imgStart.cols-SIFT_IMG_BORDER
            || yo >= images.imgStart.rows-SIFT_IMG_BORDER)
    {
        return;
    }

    for(int olayer = 1; olayer <= LAYERS; ++olayer)
    {
        int x = xo;
        int y = yo;
        int layer = olayer;

        int lx = x - x_tile;
        int ly = y - y_tile;

        //fast check to eliminate most candidates
        float& val = buffer[bufferOffset][1][1][1];
        val = sbuffer[layer][ly][lx];

        if(fabsf(val) < threshold){
            continue;
        }

        loadBufferShared<TILE_W,TILE_H,LAYERS>(sbuffer,buffer[bufferOffset],lx,ly,layer);

        //find maximum and minimum value in a 3x3x3 neighborhood
        float minN,maxN;
        findMinMax(buffer[bufferOffset],minN,maxN);

        if( !(val > 0 && val >= maxN) && !(val < 0 && val <= minN)){
            continue;
        }

        float xi=0, xr=0, xc=0, contr=0;
        int i = 0;

#ifdef SIFT_DO_SUBPIXEL_INTERPOLATION
        for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
        {
            float derivs2[6]; //xx,yy,ss,xy,xs,ys
            deriveSecond(buffer[bufferOffset],derivs2);

            float3 dD;
            derive(buffer[bufferOffset],dD);

            //direct analytical solution of the 3x3 linear system
            //this is faster than the invert+multMV from above
            float3 X;
            solveSymmetric(
                        derivs2[0], derivs2[3], derivs2[4], derivs2[1], derivs2[5], derivs2[2],
                    dD.x, dD.y, dD.z,
                    X.x, X.y, X.z
                    );

            xi = -X.z;
            xr = -X.y;
            xc = -X.x;

            if( fabsf(xi) < 0.5f && fabsf(xr) < 0.5f && fabsf(xc) < 0.5f )
                break;

            if( fabsf(xi) > (float)(INT_MAX/3) ||
                    fabsf(xr) > (float)(INT_MAX/3) ||
                    fabsf(xc) > (float)(INT_MAX/3) ){
                i = 110000;
                break;
            }

            x += Saiga::iRound(xc);
            y += Saiga::iRound(xr);
            layer += Saiga::iRound(xi);

            if (layer < 1 || layer > LAYERS ||
                    x < SIFT_IMG_BORDER || x >= images.imgStart.cols - SIFT_IMG_BORDER  ||
                    y < SIFT_IMG_BORDER || y >= images.imgStart.rows - SIFT_IMG_BORDER ){
                i = 1234124527;
                break;
            }

            //reload buffer
            lx = x - x_tile;
            ly = y - y_tile;
            loadBufferSharedMixed<TILE_W,TILE_H,LAYERS>(images,sbuffer,buffer[bufferOffset],lx,ly,layer,x_tile,y_tile);
        }

        // ensure convergence of interpolation
        if( i >= SIFT_MAX_INTERP_STEPS ){
            continue;
        }

#endif
        {
            float3 dD;
            derive(buffer[bufferOffset],dD);

            const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);

            float t = xc * dD.x + xr * dD.y + xi * dD.z;
            contr = buffer[bufferOffset][1][1][1] * img_scale + t * 0.5f;
            if( fabsf( contr ) * LAYERS < contrastThreshold)
                continue;
        }

        {
            //check edge threshold
            //principal curvatures are computed using the trace and det of Hessian
            float derivs2[6]; //xx,yy,ss,xy,xs,ys
            deriveSecond(buffer[bufferOffset],derivs2);

            float tr = derivs2[0] + derivs2[1];
            float det = derivs2[0] * derivs2[1] - derivs2[3] * derivs2[3];

            if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
                continue;
        }

        unsigned int idx = atomicInc(pointCounter, 0x7fffffff);
        if(idx < maxFeatures)
        {
            //SiftPoint& sp = keypoints[idx];
            SiftPoint sp;
            sp.ixpos = x;
            sp.iypos = y;
            sp.xpos = (x + xc)  * (1 << octave);
            sp.ypos = (y + xr)  * (1 << octave);
            sp.packOctave(octave,layer);
            sp.size = sigma * powf(2.f, float(layer + xi) / LAYERS)*(1 << octave) * 2;
            sp.response = fabsf(contr);
            sp.orientation = 0;
            Saiga::CUDA::vectorCopy(&sp, keypoints.data() + idx);
        }else{
            atomicDec(pointCounter, 0x7fffffff);
        }
    }
}

template<int OCTAVE_LAYERS>
void findPointsCaller(
        Saiga::ImageArrayView<float> images,
        Saiga::array_view<SiftPoint> keypoints,
        unsigned int* pointCounter,
        float contrastThreshold, float edgeThreshold, int octave, float sigma, int maxFeatures, int threshold)
{
    int w = images[0].cols;
    int h = images[0].rows;

    const int TILE_W = 32;
    const int TILE_H = 16;
    //        const int TILE_D = 1;
    const int RADIUS = 1;
    dim3 blocks(
                Saiga::iDivUp(w - 2 * SIFT_IMG_BORDER, TILE_W - 2 * RADIUS),
                Saiga::iDivUp(h - 2 * SIFT_IMG_BORDER, TILE_H - 2 * RADIUS),
                1
                );
    dim3 threads(TILE_W, TILE_H, 1);

    //launching a kernel with 0 blocks leads to errors in cuda 9
    if(blocks.x == 0 || blocks.y == 0)
        return;


    d_FindPointsMulti4<TILE_W, TILE_H, OCTAVE_LAYERS> << <blocks, threads >> >(images,
                                                                               keypoints,
                                                                               pointCounter,
                                                                               contrastThreshold, edgeThreshold, octave, sigma, maxFeatures, threshold);
    CUDA_SYNC_CHECK_ERROR();
}


void FindPointsMulti(Saiga::array_view<SiftPoint> keypoints, Saiga::ImageArrayView<float> images,
                     unsigned int* pointCounter,
                     float contrastThreshold, float edgeThreshold, int octave, int layers, float sigma, int maxFeatures)
{
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("SIFT_CUDA::FindPointsMulti");
#endif
    int threshold = Saiga::iFloor(0.5 * contrastThreshold / layers * 255 * SIFT_FIXPT_SCALE);

    {
        using FindKeypointsFunctionType = std::function<void(Saiga::ImageArrayView<float>, Saiga::array_view<SiftPoint>, unsigned int*, float, float, int octave, float, int, int)>;
        FindKeypointsFunctionType f[6] = {
            findPointsCaller<1>,
            findPointsCaller<2>,
            findPointsCaller<3>,
            findPointsCaller<4>,
            findPointsCaller<5>,
            findPointsCaller<6>,
        };

        f[layers - 1](images,
                      keypoints,
                      pointCounter,
                      contrastThreshold, edgeThreshold, octave, sigma, maxFeatures, threshold);
    }

    CUDA_SYNC_CHECK_ERROR();
}

}
