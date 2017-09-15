/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/filter.h"

namespace Saiga {
namespace CUDA {


//nvcc $CPPFLAGS -ptx x -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr difference.cu

template<int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__
static void d_subtract(ImageView<float> src1, ImageView<float> src2, ImageView<float> dst)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * (BLOCK_H*ROWS_PER_THREAD) + ty;


    if(x >= dst.width)
        return;

#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y += BLOCK_H){
        if(y < dst.height){
            dst(y,x) = src1(y,x) - src2(y,x);
        }
    }
}


void subtract(ImageView<float> src1, ImageView<float> src2, ImageView<float> dst){
    SAIGA_ASSERT(src1.width == dst.width && src1.height == dst.height);

    const int ROWS_PER_THREAD = 2;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst.width;
    int h = dst.height;//iDivUp(dst.height,ROWS_PER_THREAD);
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

    int height = dst.imgStart.height;

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
    //    SAIGA_ASSERT(src1.width == dst.width && src1.height == dst.height);

    SAIGA_ASSERT(src.n == dst.n + 1);
    const int ROWS_PER_THREAD = 2;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst[0].width;
    int h = dst[0].height;
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H * ROWS_PER_THREAD));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_subtractMulti<float,BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src,dst);
}




template<int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__
static void d_fill(ImageView<float> img, int h, float value)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int x = blockIdx.x*BLOCK_W + tx;
    int y = blockIdx.y*BLOCK_H + ty;

    if(x >= img.width)
        return;

    //process a fixed number of elements per thread to maximise instruction level parallelism
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y+=h){
        if(y < img.height)
            img(y,x) = value;
    }
}

void fill(ImageView<float> img, float value){
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = img.width;
    int h = iDivUp(img.height,ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_fill<BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(img,h,value);
}

template<int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__
static void d_scaleDown2EveryOther(ImageView<float> src, ImageView<float> dst, int h)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x*BLOCK_W + tx;
    int y = blockIdx.y*BLOCK_H + ty;


    if(x >= dst.width)
        return;

#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y+=h){
        if(y < dst.height){
            dst(y,x) = src(y*2,x*2);
        }
    }

}


void scaleDown2EveryOther(ImageView<float> src, ImageView<float> dst){
    SAIGA_ASSERT(src.width/2 == dst.width && src.height/2 == dst.height);
    const int ROWS_PER_THREAD = 2;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst.width;
    int h = iDivUp(dst.height,ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_scaleDown2EveryOther<BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src,dst,h);
}


#define USE_HARDWARE_INTER

#ifdef USE_HARDWARE_INTER
static texture<float, cudaTextureType2D, cudaReadModeElementType> floatTex;
#endif

template<int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__
static void d_scaleUp2Linear(ImageView<float> src, ImageView<float> dst, int h, double scale_x, double scale_y)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x*BLOCK_W + tx;
    int y = blockIdx.y*BLOCK_H + ty;

    if(x >= dst.width)
        return;

#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y+=h){
        if(y < dst.height){
#ifdef USE_HARDWARE_INTER
            //use hardware bil. interpolation
            float xf = (float(x) + 0.5f) * scale_x;
            float yf = (float(y) + 0.5f) * scale_y;
            dst(y,x) = tex2D(floatTex,xf,yf);
#else
            //software bil. interpolation
            float xf = (float(x) + 0.5f) * scale_x - 0.5f;
            float yf = (float(y) + 0.5f) * scale_y - 0.5f;
            dst(y,x) = src.inter(yf,xf);
#endif

        }
    }

}


void scaleUp2Linear(ImageView<float> src, ImageView<float> dst){
    SAIGA_ASSERT(src.width*2 == dst.width && src.height*2 == dst.height);

#ifdef USE_HARDWARE_INTER
    textureReference& floatTexRef = floatTex;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
    size_t offset;
    SAIGA_ASSERT(src.pitchBytes % 256 == 0);
    CHECK_CUDA_ERROR(cudaBindTexture2D(&offset, &floatTexRef, src.data, &desc, src.width, src.height, src.pitchBytes));
    SAIGA_ASSERT(offset == 0);
    floatTexRef.addressMode[0] = cudaAddressModeClamp;
    floatTexRef.addressMode[1] = cudaAddressModeClamp;
    floatTexRef.filterMode = cudaFilterModeLinear;
    floatTexRef.normalized = false;
#endif



    double inv_scale_x = (double)dst.width/src.width;
    double inv_scale_y = (double)dst.height/src.height;
    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;


    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst.width;
    int h = iDivUp(dst.height,ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_scaleUp2Linear<BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src,dst,h,scale_x,scale_y);
}

thrust::device_vector<float>  createGaussianBlurKernel(int radius, float sigma){
    SAIGA_ASSERT(radius <= MAX_RADIUS && radius > 0);
    const int ELEMENTS = radius * 2 + 1;
    thrust::host_vector<float> kernel(ELEMENTS);
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*sigma*sigma);
    for (int j=-radius;j<=radius;j++) {
        kernel[j+radius] = (float)expf(-(double)j*j*ivar2);
        kernelSum += kernel[j+radius];
    }
    for (int j=-radius;j<=radius;j++)
        kernel[j+radius] /= kernelSum;
    return thrust::device_vector<float>(kernel);
}

void applyFilterSeparateSinglePass(ImageView<float> src, ImageView<float> dst, array_view<float> kernel){
    int radius = kernel.size()/2;
    //inner 75 is the fastest for small kernels
    if(radius < 7)
    {
        convolveSinglePassSeparateInner75(src,dst,kernel,radius);
    }else
    {
        convolveSinglePassSeparateOuterHalo(src,dst,kernel,radius);
    }
}



__constant__ float d_Kernel[MAX_RADIUS*2+1];


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void d_convolveOuterLinear(ImageView<T> src, ImageView<T> dst)
{
    const unsigned BLOCK_H2 = BLOCK_H * Y_ELEMENTS;

    //for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ T buffer[BLOCK_H2 + 2*RADIUS][BLOCK_W + 2*RADIUS];
    //for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ T buffer2[BLOCK_H2][BLOCK_W + 2*RADIUS];
    //total s mem per block = 6400
    //with 512 threads per block smem per sm: 25600 -> 100% occ


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int t = tx + ty * BLOCK_W;
    int xp = blockIdx.x*BLOCK_W + tx;
    int yp = blockIdx.y*BLOCK_H2 + ty;


    int blockStartX = blockIdx.x*BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y*BLOCK_H2 - RADIUS;

    const int blockSizeX = BLOCK_W + 2*RADIUS;
    const int blockSizeY = BLOCK_H2 + 2*RADIUS;

    //fill buffer
    for(int i = t; i < blockSizeX * blockSizeY; i += (BLOCK_W*BLOCK_H)){
        int x = i % blockSizeX;
        int y = i / blockSizeX;
        int gx = x + blockStartX;
        int gy = y + blockStartY;
        src.clampToEdge(gy,gx);
        buffer[y][x] = src(gy,gx);
    }

    __syncthreads();


    T *kernel = d_Kernel;

    for(int i = t; i < blockSizeX * BLOCK_H2; i += (BLOCK_W*BLOCK_H)){
        int x = i % blockSizeX;
        int y = i / blockSizeX;
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer[y + RADIUS + j][x] * kernel[kernelIndex];
        }
        buffer2[y][x] = sum;
    }

    __syncthreads();

    for(int i =0; i < Y_ELEMENTS; ++i){
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if(dst.inImage(yp,xp))
            dst(yp,xp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template<typename T, int RADIUS>
inline
void convolveOuterLinear(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;

    const int BLOCK_W = 32;
    const int BLOCK_H = 16;
    const int Y_ELEMENTS = 2;
    dim3 blocks(
                Saiga::iDivUp(w, BLOCK_W ),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveOuterLinear<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void d_convolveOuterHalo(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int BLOCK_H2 = BLOCK_H * Y_ELEMENTS;
    const unsigned int WARPS_PER_BLOCK = BLOCK_W * BLOCK_H / 32; //16
    static_assert(WARPS_PER_BLOCK == 16, "warps per block wrong");



    //for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ T buffer[BLOCK_H2 + 2*RADIUS][BLOCK_W + 2*RADIUS];
    //for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ T buffer2[BLOCK_H2][BLOCK_W + 2*RADIUS];
    //total s mem per block = 6400
    //with 512 threads per block smem per sm: 25600 -> 100% occ


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int t = tx + ty * BLOCK_W;
    const unsigned int warp_lane = t / 32;
    const unsigned int lane_id = t & 31;

    int xp = blockIdx.x*BLOCK_W + tx;
    int yp = blockIdx.y*BLOCK_H2 + ty;
    int x = xp;
    int y = yp;

    const unsigned int x_tile = blockIdx.x * BLOCK_W;
//    const unsigned int y_tile = blockIdx.y * BLOCK_H2;

    int blockStartX = blockIdx.x*BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y*BLOCK_H2 - RADIUS;

    const int blockSizeX = BLOCK_W + 2*RADIUS;
//    const int blockSizeY = BLOCK_H2 + 2*RADIUS;

    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        buffer[ty + i * BLOCK_H + RADIUS][tx + RADIUS]  = src.clampedRead(y + i * BLOCK_H,x);
    }

    //top and bottom halo
    if(warp_lane < 4)
    {
        const unsigned int num_warps = 4;
        for(int i = warp_lane; i < RADIUS; i+=num_warps)
        {
            buffer[i][lane_id + RADIUS]  =
                    src.clampedRead(blockStartY + i,x_tile + lane_id);

            buffer[BLOCK_H2 + RADIUS + i][lane_id + RADIUS]  =
                    src.clampedRead(blockStartY + BLOCK_H2 + RADIUS  + i,x_tile + lane_id);
        }
    }

    const unsigned int side_halo_rows_per_warp = 32 / RADIUS;

    int local_warp_id = lane_id / RADIUS;
    int local_lane_id = lane_id % RADIUS;

    //left halo
    if(warp_lane >= 4 && warp_lane < 10)
    {
        const unsigned int num_warps = 6;
        int wid = warp_lane - 4;
        int rows = BLOCK_H2 + 2 * RADIUS;

        for(int i = wid * side_halo_rows_per_warp + local_warp_id;i < rows; i += num_warps*side_halo_rows_per_warp)
        {
            if(local_warp_id < side_halo_rows_per_warp)
            {
                buffer[i][local_lane_id]  =
                        src.clampedRead(blockStartY + i,blockStartX + local_lane_id);
            }
        }
    }

    //right halo
    if(warp_lane >= 10 && warp_lane < 16)
    {
        const unsigned int num_warps = 6;
        int wid = warp_lane - 10;
        int rows = BLOCK_H2 + 2 * RADIUS;

        for(int i = wid * side_halo_rows_per_warp + local_warp_id;i < rows; i += num_warps*side_halo_rows_per_warp)
        {
            if(local_warp_id < side_halo_rows_per_warp)
            {
                buffer[i][local_lane_id + RADIUS + BLOCK_W]  =
                        src.clampedRead(blockStartY + i,blockStartX + local_lane_id + RADIUS + BLOCK_W);
            }
        }
    }

    __syncthreads();


    T *kernel = d_Kernel;

    for(int i = t; i < blockSizeX * BLOCK_H2; i += (BLOCK_W*BLOCK_H)){
        int x = i % blockSizeX;
        int y = i / blockSizeX;
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer[y + RADIUS + j][x] * kernel[kernelIndex];
        }
        buffer2[y][x] = sum;
    }

    __syncthreads();

    for(int i =0; i < Y_ELEMENTS; ++i){
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if(dst.inImage(yp,xp))
            dst(yp,xp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template<typename T, int RADIUS>
inline
void convolveOuterHalo(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;

    const int BLOCK_W = 32;
    const int BLOCK_H = 16;
    const int Y_ELEMENTS = 2;
    dim3 blocks(
                Saiga::iDivUp(w, BLOCK_W ),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveOuterHalo<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void d_convolveInner(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int TILE_H = BLOCK_H;
    const unsigned int TILE_W = BLOCK_W;

    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
//    int t = tx + ty * BLOCK_W;

    int x_tile = blockIdx.x * (TILE_W - 2 * RADIUS) - RADIUS;
    int y_tile = blockIdx.y * (TILE_H2 - 2 * RADIUS) - RADIUS;

    int x = x_tile + tx;
    int y = y_tile + ty;


    __shared__ T buffer[TILE_H2][TILE_W];
    __shared__ T buffer2[TILE_H2 - RADIUS * 2][TILE_W];



    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
        buffer[ty + i * TILE_H][tx]  = src.clampedRead(y + i * TILE_H,x);



    __syncthreads();


    T *kernel = d_Kernel;

    //convolve along y axis
    //    if(ty > RADIUS && ty < TILE_H2 - RADIUS)
    //    {
    //        int oy = ty - RADIUS;

    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        //        int gx = x;
        //        int gy = y + i * TILE_H;
        int lx = tx;
        int ly = ty + i * TILE_H;

        if(ly < RADIUS || ly >= TILE_H2 - RADIUS)
            continue;

        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer[ly + j][lx] * kernel[kernelIndex];
        }
        buffer2[ly - RADIUS][lx] = sum;
    }



    __syncthreads();

    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        int gx = x;
        int gy = y + i * TILE_H;

        int lx = tx;
        int ly = ty + i * TILE_H;

        if(ly < RADIUS || ly >= TILE_H2 - RADIUS)
            continue;

        if(lx < RADIUS || lx >= TILE_W - RADIUS)
            continue;

        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer2[ly - RADIUS][lx + j] * kernel[kernelIndex];
        }

        //        if(dst.inImage(gx,gy))
        //            dst(g,yp) = sum;
        dst.clampedWrite(gy,gx,sum);
    }



#if 0

    for(int i =0; i < Y_ELEMENTS; ++i){
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if(dst.inImage(xp,yp))
            dst(xp,yp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
#endif
}

template<typename T, int RADIUS, bool LOW_OCC = false>
inline
void convolveInner(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;


    const int BLOCK_W = LOW_OCC ? 64 : 32;
    const int BLOCK_H = LOW_OCC ? 8 : 16;
    const int Y_ELEMENTS = LOW_OCC ? 4 : 2;
    dim3 blocks(
                Saiga::iDivUp(w, BLOCK_W - 2 * RADIUS),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS - 2 * RADIUS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveInner<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}



void convolveSinglePassSeparateOuterLinear(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveOuterLinear<float,1>(src,dst); break;
    case 2: CUDA::convolveOuterLinear<float,2>(src,dst); break;
    case 3: CUDA::convolveOuterLinear<float,3>(src,dst); break;
    case 4: CUDA::convolveOuterLinear<float,4>(src,dst); break;
    case 5: CUDA::convolveOuterLinear<float,5>(src,dst); break;
    case 6: CUDA::convolveOuterLinear<float,6>(src,dst); break;
    case 7: CUDA::convolveOuterLinear<float,7>(src,dst); break;
    case 8: CUDA::convolveOuterLinear<float,8>(src,dst); break;
    case 9: CUDA::convolveOuterLinear<float,9>(src,dst); break;
    }
}

void convolveSinglePassSeparateOuterHalo(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveOuterHalo<float,1>(src,dst); break;
    case 2: CUDA::convolveOuterHalo<float,2>(src,dst); break;
    case 3: CUDA::convolveOuterHalo<float,3>(src,dst); break;
    case 4: CUDA::convolveOuterHalo<float,4>(src,dst); break;
    case 5: CUDA::convolveOuterHalo<float,5>(src,dst); break;
    case 6: CUDA::convolveOuterHalo<float,6>(src,dst); break;
    case 7: CUDA::convolveOuterHalo<float,7>(src,dst); break;
    case 8: CUDA::convolveOuterHalo<float,8>(src,dst); break;
    }
}

void convolveSinglePassSeparateInner(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveInner<float,1>(src,dst); break;
    case 2: CUDA::convolveInner<float,2>(src,dst); break;
    case 3: CUDA::convolveInner<float,3>(src,dst); break;
    case 4: CUDA::convolveInner<float,4>(src,dst); break;
    case 5: CUDA::convolveInner<float,5>(src,dst); break;
    case 6: CUDA::convolveInner<float,6>(src,dst); break;
    case 7: CUDA::convolveInner<float,7>(src,dst); break;
    case 8: CUDA::convolveInner<float,8>(src,dst); break;
    }
}


void convolveSinglePassSeparateInner75(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveInner<float,1,true>(src,dst); break;
    case 2: CUDA::convolveInner<float,2,true>(src,dst); break;
    case 3: CUDA::convolveInner<float,3,true>(src,dst); break;
    case 4: CUDA::convolveInner<float,4,true>(src,dst); break;
    case 5: CUDA::convolveInner<float,5,true>(src,dst); break;
    case 6: CUDA::convolveInner<float,6,true>(src,dst); break;
    case 7: CUDA::convolveInner<float,7,true>(src,dst); break;
    case 8: CUDA::convolveInner<float,8,true>(src,dst); break;
    }
}

}
}


