/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/cuda/device_helper.h"

namespace Saiga {
namespace CUDA {

//nvcc $CPPFLAGS -ptx x -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr difference.cu

template<int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__
static void d_subtract(SiftImageType src1, SiftImageType src2, SiftImageType dst)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * (BLOCK_H*ROWS_PER_THREAD) + ty;


    if(x >= dst.cols)
        return;

#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y += BLOCK_H){
        if(y < dst.rows){
            dst(y,x) = src1(y,x) - src2(y,x);
        }
    }
}


void subtract(SiftImageType src1, SiftImageType src2, SiftImageType dst){
    SAIGA_ASSERT(src1.cols == dst.cols && src1.rows == dst.rows);

    const int ROWS_PER_THREAD = 2;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst.cols;
    int h = dst.rows;//iDivUp(dst.rows,ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H * ROWS_PER_THREAD));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_subtract<BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src1,src2,dst);
}


template<typename T, int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ void d_subtractMulti(
        ImageArrayView<float> src, ImageArrayView<float> dst)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int x = blockIdx.x * BLOCK_W + tx;
    int ys = blockIdx.y * (BLOCK_H*ROWS_PER_THREAD) + ty;

    int height = dst.imgStart.rows;

    if(!src.imgStart.inImage(ys,x))
        return;

    T lastVals[ROWS_PER_THREAD];


    int y = ys;
#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y+=BLOCK_H){
        if(y < height){
            lastVals[i] = src.atIARVxxx(0,y,x);
        }
    }

    for(int i = 0; i < dst.n; ++i){
        int y = ys;
#pragma unroll
        for(int j = 0; j < ROWS_PER_THREAD; ++j, y+=BLOCK_H){
            if(y < height){
                T nextVal = src.atIARVxxx(i+1,y,x);
                dst.atIARVxxx(i,y,x) = nextVal - lastVals[j];
                lastVals[j] = nextVal;
            }
        }
    }
}

void subtractMulti(ImageArrayView<float> src, ImageArrayView<float> dst){
    //    SAIGA_ASSERT(src1.cols == dst.cols && src1.rows == dst.rows);

    SAIGA_ASSERT(src.n == dst.n + 1);
    const int ROWS_PER_THREAD = 2;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst[0].cols;
    int h = dst[0].rows;
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H * ROWS_PER_THREAD));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_subtractMulti<float,BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src,dst);
}


}
}


