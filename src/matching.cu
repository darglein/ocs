/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "matching.h"

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/reduce.h"

namespace cudasift {

template<unsigned int BLOCK_W2, unsigned int BLOCK_H, unsigned int LOCAL_WARP_W=1, unsigned int DESCRIPTOR_SIZE2=128>
__global__ static
__launch_bounds__(1024,2)
void d_computeDistanceMatrix2(Saiga::ImageView<float> distances,
                              float *descY2, float *descX2)
{
    const unsigned int BLOCK_W = BLOCK_W2 / LOCAL_WARP_W;
    const unsigned int DESCRIPTOR_SIZE = DESCRIPTOR_SIZE2 / 2;
    float2* descX = reinterpret_cast<float2*>(descX2);
    float2* descY = reinterpret_cast<float2*>(descY2);

    //Memory usage:
    //32x32 blocks: 2 * 32 * 128 * sizeof(float) = 32768 -> 1 block per sm (50% occupancy)
    //24x24 blocks: 2 * 32 * 128 * sizeof(float) = 24576 -> 2 block per sm (56.25% occupancy)
    //32x16 blocks:  (32+16) * 128 * sizeof(float) = 24576 -> 2 blocks per sm (50% occupancy)
    //16x16 blocks:  (16+16) * 128 * sizeof(float) = 16384 -> 3 blocks per sm (37.5% occupancy)
    __shared__ float2 siftPointsX[BLOCK_W][DESCRIPTOR_SIZE];
    __shared__ float2 siftPointsY[BLOCK_H][DESCRIPTOR_SIZE];

    const int lane = threadIdx.x & (LOCAL_WARP_W - 1);
    const int tx = threadIdx.x / LOCAL_WARP_W;
    const int ty = threadIdx.y;
    const int t = ty * BLOCK_W2 + threadIdx.x;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;


    int sx = blockIdx.x * BLOCK_W;
    int sy = blockIdx.y * BLOCK_H;


    //copy descriptors to shared memory
#pragma unroll
    WARP_FOR_NO_IF(i,t,BLOCK_W * DESCRIPTOR_SIZE,BLOCK_W2*BLOCK_H){
        int descId = i / DESCRIPTOR_SIZE;
        int elId = i % DESCRIPTOR_SIZE;
        auto globalId = sx * DESCRIPTOR_SIZE + i;
        siftPointsX[descId][elId] = globalId < distances.width*DESCRIPTOR_SIZE ? descX[globalId] : float2();
    }

#pragma unroll
    WARP_FOR_NO_IF(i,t,BLOCK_H * DESCRIPTOR_SIZE,BLOCK_W2*BLOCK_H){
        int descId = i / DESCRIPTOR_SIZE;
        int elId = i % DESCRIPTOR_SIZE;
        auto globalId = sy * DESCRIPTOR_SIZE + i;
        siftPointsY[descId][elId] = globalId < distances.height*DESCRIPTOR_SIZE ? descY[globalId] : float2();
    }

    __syncthreads();

    float sum = 0.0f;

    WARP_FOR_NO_IF(i,lane,DESCRIPTOR_SIZE,LOCAL_WARP_W){
        int itx = (i + tx * LOCAL_WARP_W + ty * BLOCK_W2) & (DESCRIPTOR_SIZE - 1);
        float2 px = siftPointsX[tx][itx];
        float2 py = siftPointsY[ty][itx];
        auto tmp1 = px.x - py.x;
        auto tmp2 = px.y - py.y;
        sum += tmp1 * tmp1 + tmp2 * tmp2;
    }

    sum = Saiga::CUDA::warpReduceSum<float,LOCAL_WARP_W,false>(sum);
    sum = sqrtf(sum);

    if (lane == 0 && y<distances.height){
        distances(y,x) = (x<distances.width? sum : 45647431.0f);
    }
}

template<unsigned int THREADS_PER_BLOCK, unsigned int K>
__global__ static
void sortK(float *corrData, Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index, int numPts1, int corrWidth, int corrPitch)
{
    __shared__ float maxScore[THREADS_PER_BLOCK * K]; //2048 * 4 = 8100
    __shared__ int maxIndex[THREADS_PER_BLOCK * K];

//    Saiga::CUDA::ThreadInfo<THREADS_PER_BLOCK> ti;

    int local_thread_id = threadIdx.x;
    int block_id        = blockIdx.x;

    int thread_id       = THREADS_PER_BLOCK * block_id + local_thread_id;

    int id = thread_id;
    int idx = local_thread_id;

    if(id >= numPts1)
        return;


    float *corrs = corrData + id * corrPitch;

    float highestVal = 4326626;

    for(int k = 0 ; k < K ; ++k){
        maxScore[idx * K + k] = 4326626;
    }

    for (int i = 0; i < corrWidth; i += 1) {
        int cid = i;
        float val = corrs[cid];
        if (val >= highestVal){
            continue;
        }

        float lastV;
        int lastIdx;

        //insertion sort
        for(int k = 0 ; k < K ; ++k){
            auto v = maxScore[idx * K + k];
            if(val < v){
                lastV = v;
                lastIdx = maxIndex[idx * K + k];

                maxScore[idx * K + k] = val;
                maxIndex[idx * K + k] = cid;

                cid = lastIdx;
                val = lastV;
            }
        }

        highestVal = maxScore[idx * K + K - 1];

    }

    __syncthreads();


    for(int k = 0 ; k < K ; ++k){
        out_distance[id * K + k] = maxScore[idx * K + k];
        out_index[id * K + k] = maxIndex[idx * K + k];
    }

}


template<typename T, typename Ti, unsigned int LOCAL_WARP_SIZE=32, bool RESULT_FOR_ALL_THREADS=false>
__device__ inline
void warpReduceMinIndex(T& val, Ti& index) {
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE/2; offset > 0; offset /= 2){
        auto v = RESULT_FOR_ALL_THREADS ? __shfl_xor(val, offset) : __shfl_down(val, offset);
        auto i = RESULT_FOR_ALL_THREADS ? __shfl_xor(index, offset) : __shfl_down(index, offset);
        val = min(val , v);
        index = (v == val) ? i : index;
    }
}




template <typename T> HD inline void swap ( T& a, T& b )
{
    T c(a); a=b; b=c;
}

template<unsigned int THREADS_PER_BLOCK, unsigned int LOCAL_WARP_SIZE, unsigned int K>
__global__ static
void kmin( Saiga::ImageView<float> distances, float* outData, int* outIndices)
{
    const int warps_per_block = THREADS_PER_BLOCK / LOCAL_WARP_SIZE;

    //for every warp K elements
    //memory for K = 4 and warpsize = 32:
    //2 * 4 * 4 * 16 * sizeof(float) = 2048 byte
    __shared__ float minValuesRes[warps_per_block][K];
    __shared__ volatile int minIndicesRes[warps_per_block][K];


//    Saiga::CUDA::ThreadInfo<THREADS_PER_BLOCK,LOCAL_WARP_SIZE> ti;


//    warp_id         = thread_id   / LOCAL_WARP_SIZE;

    int  local_thread_id = threadIdx.x;
    int warp_lane       = local_thread_id / LOCAL_WARP_SIZE;
    int tid = local_thread_id + blockIdx.x * THREADS_PER_BLOCK;
    int lane_id = local_thread_id & (LOCAL_WARP_SIZE-1);
    int y = tid / LOCAL_WARP_SIZE;


    if(y >= distances.height)
        return;

    float minValues[K];
    int minIndices[K];

    //    float& lowestValue = minValues[0];
    float& highestValue = minValues[K-1];
    int& highestIndex = minIndices[K-1];

#pragma unroll
    for(int k = 0 ; k < K ; ++k){
        minValues[k] = 4326626;
    }

    //    data += y * pitch;    //for every element
    for (int i = lane_id; i < distances.width; i += LOCAL_WARP_SIZE) {
        int newIdx = i;
        float newval = distances(y,i);
        //ignore all values that are larger than the largest value
        if (newval >= highestValue){
            continue;
        }

        //the new value must be insertet into our local list
        highestValue = newval;
        highestIndex = newIdx;

        //do a reverse 1 element bubble sort from the top to the bottom
#pragma unroll
        for(int k = K - 1; k > 0; --k){
            if(minValues[k] < minValues[k-1]){
                //swap
                swap(minValues[k],minValues[k-1]);
                swap(minIndices[k],minIndices[k-1]);
            }else{
                break;
            }
        }

    }


    //now every thread has 4 potential min candidates
    //the overall minimum can be found with a reduction on the first variable

    int currentMinPtr = 0;

#pragma unroll
    for(int k = 0 ; k < K ; ++k){

        float currentMinValue;
        int currentMinIndex;
#pragma unroll
        for(int j = 0 ; j < K ; ++j){
            //similar to shuffle copy use this trick to prevent compiler constant propagation
            if(j <= currentMinPtr){
                currentMinValue = minValues[j];
                currentMinIndex = minIndices[j];
            }
        }

        auto vTmp = currentMinValue;
        auto iTmp = currentMinIndex;
        //we can't use xor shuffle broadcast here because the order is important to obtain the same index
        //when the value is the same
        warpReduceMinIndex<float,int,LOCAL_WARP_SIZE,false>(vTmp,iTmp);

        if(lane_id == 0){
            minValuesRes[warp_lane][k] = vTmp;
            minIndicesRes[warp_lane][k] = iTmp;
        }
        iTmp = minIndicesRes[warp_lane][k];

        if(iTmp == currentMinIndex){
            currentMinPtr++;
        }
    }



    outData += y * K;
    outIndices += y * K;
    for(int k = lane_id ; k < K ; k+=LOCAL_WARP_SIZE){
        outData[k] = minValuesRes[warp_lane][k];
        outIndices[k] = minIndicesRes[warp_lane][k];
    }


}

MatchGPU::MatchGPU(int nfeatures) : maxPoints(nfeatures){
    initMemory();
}

void MatchGPU::initMemory(){
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("initMemory");
#endif
    distances = Saiga::CUDA::CudaImage<float>(maxPoints,maxPoints, Saiga::iDivUp(maxPoints, 16)*16*sizeof(float));
}

void MatchGPU::computeDistanceMatrix(Saiga::array_view<float> descriptors1, Saiga::array_view<float> descriptors2){

#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("computeDistanceMatrix");
#endif


    {
        const int BLOCK_W = 32;
        const int BLOCK_H = 32;
        const int LOCAL_WARP_W = 1;
        cudaFuncSetSharedMemConfig(d_computeDistanceMatrix2<BLOCK_W,BLOCK_H,LOCAL_WARP_W>,cudaSharedMemBankSizeEightByte);

        dim3 threads(BLOCK_W, BLOCK_H);
        dim3 blocks(Saiga::iDivUp(distances.width,BLOCK_W/LOCAL_WARP_W), Saiga::iDivUp(distances.height, BLOCK_H));
        d_computeDistanceMatrix2<BLOCK_W,BLOCK_H,LOCAL_WARP_W><<<blocks, threads>>>(distances,descriptors1.data(), descriptors2.data());
    }

    CUDA_SYNC_CHECK_ERROR();
}

template<int K>
static void computeKNN2(Saiga::ImageView<float> distances, Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index){
    const int THREADS_PER_ROW = 32;
    const int BLOCK_SIZE = 128;
    int blocks = Saiga::iDivUp(distances.height*THREADS_PER_ROW,BLOCK_SIZE);
    kmin<BLOCK_SIZE,THREADS_PER_ROW,K><<<blocks,BLOCK_SIZE>>>(distances,
                                                              thrust::raw_pointer_cast(out_distance.data()),
                                                              thrust::raw_pointer_cast(out_index.data())
                                                              );
}

void MatchGPU::computeKNN(Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index, int k){

    SAIGA_ASSERT(k > 0 && k <= 4);

#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("computeKNN");
#endif

    switch(k){
    case 1:
        computeKNN2<1>(distances,out_distance,out_index);
        break;
    case 2:
        computeKNN2<2>(distances,out_distance,out_index);
        break;
    case 3:
        computeKNN2<3>(distances,out_distance,out_index);
        break;
    case 4:
        computeKNN2<4>(distances,out_distance,out_index);
        break;
    }

}

void MatchGPU::knnMatch(Saiga::array_view<float> descriptors1, Saiga::array_view<float> descriptors2, Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index, int k)
{
    int numPoints1 = descriptors1.size() / 128;
    int numPoints2 = descriptors2.size() / 128;
    if (numPoints1 == 0 || numPoints2 == 0)
        return;

    distances.height = numPoints1;
    distances.width = numPoints2;

    SAIGA_ASSERT(out_distance.size() >= numPoints1 * k);

    computeDistanceMatrix(descriptors1,descriptors2);
    computeKNN(out_distance,out_index,k);

    CUDA_SYNC_CHECK_ERROR();
}		 





template<unsigned int THREADS_PER_BLOCK, unsigned int LOCAL_WARP_SIZE>
__global__ static
void d_filterByRadius(Saiga::array_view<SiftPoint> keypoints1, Saiga::array_view<SiftPoint> keypoints2,
                      Saiga::ImageView<float> distances,
                      float radius)
{
//    Saiga::CUDA::ThreadInfo<THREADS_PER_BLOCK,LOCAL_WARP_SIZE> ti;
    int local_thread_id = threadIdx.x;
    int block_id        = blockIdx.x;

    int lane_id         = local_thread_id & (LOCAL_WARP_SIZE-1);
    int thread_id       = THREADS_PER_BLOCK * block_id + local_thread_id;
    int warp_id         = thread_id   / LOCAL_WARP_SIZE;

    int y = warp_id;

    if(y >= distances.height)
        return;

    float r2 = radius * radius;

    float2 pos = reinterpret_cast<float2*>((keypoints1.data()+y))[0];

    for(int i = lane_id; i < distances.width; i += LOCAL_WARP_SIZE){
        static_assert(sizeof(SiftPoint) == 32, "siftpoint size");
        float2 pos2 = reinterpret_cast<float2*>((keypoints2.data()+i))[0];
        float dx = pos2.x - pos.x;
        float dy = pos2.y - pos.y;
        float d2 = dx * dx + dy * dy;
        if(d2 > r2){
            distances(y,i) = 36464067;
        }
    }
}


void MatchGPU::filterByRadius(Saiga::array_view<SiftPoint> keypoints1, Saiga::array_view<SiftPoint> keypoints2, float r){
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("filterByRadius");
#endif
    const int THREADS_PER_ROW = 32;
    const int BLOCK_SIZE = 128;
    int blocks = Saiga::iDivUp(distances.height*THREADS_PER_ROW,BLOCK_SIZE);
    d_filterByRadius<BLOCK_SIZE,THREADS_PER_ROW><<<blocks,BLOCK_SIZE>>>(keypoints1,keypoints2,distances,r);
    CUDA_SYNC_CHECK_ERROR();
}

void MatchGPU::radiusMatch(Saiga::array_view<SiftPoint> keypoints1, Saiga::array_view<float> descriptors1, Saiga::array_view<SiftPoint> keypoints2, Saiga::array_view<float> descriptors2, Saiga::array_view<float> out_distance, Saiga::array_view<int> out_index, int k, float r)
{
    int numPoints1 = keypoints1.size();
    int numPoints2 = keypoints2.size();
    if (numPoints1 == 0 || numPoints2 == 0)
        return;

    distances.height = numPoints1;
    distances.width = numPoints2;

    SAIGA_ASSERT(out_distance.size() >= numPoints1 * k);

    computeDistanceMatrix(descriptors1,descriptors2);
    filterByRadius(keypoints1,keypoints2,r);
    computeKNN(out_distance,out_index,k);

    CUDA_SYNC_CHECK_ERROR();
}

}
