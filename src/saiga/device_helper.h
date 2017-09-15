/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/common.h"

#if !defined(IS_CUDA)
#error device_helper.h must only be included by nvcc
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//atomicAdd is already defined for compute capability 6.x and higher.
#else
#if 0
__device__ inline
double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline
double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
#endif


//CUDA_ASSERT

#if defined(CUDA_DEBUG)


namespace Saiga {
namespace CUDA {
__device__ inline
void cuda_assert_fail (const char *__assertion, const char *__file,
               unsigned int __line, const char *__function){
    printf("Assertion '%s' failed!\n"
           "  File: %s:%d\n"
           "  Function: %s\n"
           "  Thread: %d,%d,%d\n"
           "  Block: %d, %d, %d\n",
           __assertion, __file, __line, __function,
           threadIdx.x,threadIdx.y,threadIdx.z,
           blockIdx.x,blockIdx.y,blockIdx.z);
    //provoke a segfault
     *(int*)0 = 0;
}
}
}

# define CUDA_ASSERT(expr)							\
  ((expr)								\
   ? static_cast<void>(0)						\
   : Saiga::CUDA::cuda_assert_fail (#expr, __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION))

#else

# define CUDA_ASSERT(expr)		( static_cast<void>(0))

#endif


#define WARP_FOR_NO_IF(_variableName, _initExpr, _length, _step) for (unsigned int _variableName=_initExpr, _k=0; _k < Saiga::iDivUp(_length,_step);_k++, _variableName+=_step)

#define WARP_FOR(_variableName, _initExpr, _length, _step) WARP_FOR_NO_IF(_variableName, _initExpr, _length, _step) \
    if(_variableName < _length)



namespace Saiga {
namespace CUDA{

/**
 * A simple helper struct that can be used at the beginning of each kernel to compute some usefull variables.
 * Don't worry about variables that you are not going going to use, because they are optimized away :).
 *
 * If you know the number of threads per block at compile time pass it as the first template argument.
 * LOCAL_WARP_SIZE can be used for example in partial warp reductions.
 * LOCAL_WARP_SIZE must be one of these values: 1,2,4,8,16,32
 *
 * Note: In some libraries you will find the __mul24 for these index computations, but from cuda 8 api:
 *  "_[u]mul24 are legacy intrinsic functions that have no longer any reason to be used"
 *
 * Note: It is important to use unsigend ints here, because then the compiler can replace expensive integer divisions
 * with fast bit shifts. The same counts when accessing these variables. Either use "auto" or "unsigend int".
 */
template<unsigned int THREADS_PER_BLOCK = 0, unsigned int LOCAL_WARP_SIZE=WARP_SIZE>
struct ThreadInfo{

    unsigned int local_thread_id;   // local thread id in the block PTX: %tid.x
    unsigned int thread_id;         // global thread index
    unsigned int block_id;          // id of this block PTX: %ctaid.x
    unsigned int lane_id;           // thread index within the warp
    unsigned int warp_id;           // global warp index

    unsigned int threads_per_block;
    unsigned int warp_lane;         // warp index within the block
    unsigned int num_blocks;        //number of blocks in that grid PTX: %nctaid.x
    unsigned int num_warps_block;   // total number of active warps
    unsigned int num_warps;         // total number of active warps
    unsigned int grid_size;         //total number of threads in the grid

    __device__
    ThreadInfo(){
        if(THREADS_PER_BLOCK > 0){
            threads_per_block = THREADS_PER_BLOCK;
        }else{
            threads_per_block = blockDim.x;
        }
        local_thread_id = threadIdx.x;
        block_id        = blockIdx.x;
        num_blocks      = gridDim.x;

        //Note: Beccause of this line LOCAL_WARP_SIZE must be a power of 2
        lane_id         = local_thread_id & (LOCAL_WARP_SIZE-1);
        thread_id       = threads_per_block * block_id + local_thread_id;

        grid_size       = num_blocks * threads_per_block;

        warp_id         = thread_id   / LOCAL_WARP_SIZE;
        warp_lane       = local_thread_id / LOCAL_WARP_SIZE;
        num_warps_block = threads_per_block / LOCAL_WARP_SIZE;
        num_warps       = num_warps_block * num_blocks;
    }
};

}
}


namespace Saiga {
namespace CUDA{

/**
 *  Copy a large datatype T with vector instructions.
 * Example: if you have a 32 byte datatype and use int4 as a vector type
 * the compiler generates two 16 byte load instructions, instead of potentially
 * eight 4 byte loads.
 */
template<typename T, typename VECTOR_TYPE=int4>
__device__ inline
void vectorCopy(const T* source, T* dest){
    static_assert(sizeof(T) % sizeof(VECTOR_TYPE) == 0, "Incompatible types.");
    const VECTOR_TYPE* source4 = reinterpret_cast<const VECTOR_TYPE*>(source);
    VECTOR_TYPE* dest4 = reinterpret_cast<VECTOR_TYPE*>(dest);
#pragma unroll
    for(int i = 0; i < sizeof(T)/sizeof(VECTOR_TYPE); ++i ){
        dest4[i] = source4[i];
    }
}


/**
 * Copy an array of small T's with vector instructions.
 */
template<typename T, typename VECTOR_TYPE=int4>
__device__ inline
void vectorArrayCopy(const T* source, T* dest){
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0, "Wrong use of this function.");
    static_assert(sizeof(VECTOR_TYPE) >= sizeof(T), "Wrong use of this function.");
    reinterpret_cast<VECTOR_TYPE*>( dest )[0] = reinterpret_cast<const VECTOR_TYPE*>( source )[0];
}

//enum CacheLoadModifier
//{
//    LOAD_DEFAULT,       ///< Default (no modifier)
//    LOAD_CA,            ///< Cache at all levels
//    LOAD_CG,            ///< Cache at global level
//    LOAD_CS,            ///< Cache streaming (likely to be accessed once)
//    LOAD_CV,            ///< Cache as volatile (including cached system lines)
//    LOAD_LDG,           ///< Cache as texture
//    LOAD_VOLATILE,      ///< Volatile (any memory space)
//};

//_CUB_LOAD_ALL(LOAD_CA, ca)
//_CUB_LOAD_ALL(LOAD_CG, cg)
//_CUB_LOAD_ALL(LOAD_CS, cs)
//_CUB_LOAD_ALL(LOAD_CV, cv)
//_CUB_LOAD_ALL(LOAD_LDG, global.nc)

//#define _CUB_LOAD_4(cub_modifier, ptx_modifier)

/**
 * Cache at global level (cache in L2 and below, not L1).
 * Use ld.cg to cache loads only globally, bypassing the L1 cache, and cache only in the L2 cache.
 *
 * Read more at: http://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators
 * Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
 */
__device__ __forceinline__
unsigned int loadNoL1Cache4(unsigned int const *ptr){
    unsigned int retval;
    asm volatile ("ld.cg.u32 %0, [%1];" :
                  "=r"(retval) :
                  "l"(ptr));
    return retval;
}

__device__ __forceinline__
uint2 loadNoL1Cache8(uint2 const *ptr){
    uint2 retval;
    asm volatile ("ld.cg.v2.u32 {%0, %1}, [%2];" :
                  "=r"(retval.x),
                  "=r"(retval.y) :
                  "l"(ptr));
    return retval;
}

__device__ __forceinline__
uint4 loadNoL1Cache16(uint4 const *ptr){
    uint4 retval;
    asm volatile ("ld.cg.v4.u32 {%0, %1, %2, %3}, [%4];" :
                  "=r"(retval.x),
                  "=r"(retval.y),
                  "=r"(retval.z),
                  "=r"(retval.w):
                  "l"(ptr));
    return retval;
}

template<typename T>
__device__ inline
T loadNoL1Cache(T const *ptr){
    T t;
    if(sizeof(T)==4)
        reinterpret_cast<unsigned int*>(&t)[0] = loadNoL1Cache4(reinterpret_cast<unsigned int const *>(ptr));
    if(sizeof(T)==8)
        reinterpret_cast<uint2*>(&t)[0] = loadNoL1Cache8(reinterpret_cast<uint2 const *>(ptr));
    if(sizeof(T)==16)
        reinterpret_cast<uint4*>(&t)[0] = loadNoL1Cache16(reinterpret_cast<uint4 const *>(ptr));
    return t;
}



//B.10. Read-Only Data Cache Load Function
template<typename T>
__device__ inline
T ldg(const T* address){
#if __CUDA_ARCH__ >= 350
    //The read-only data cache load function is only supported by devices of compute capability 3.5 and higher.
    return __ldg(address);
#else
    return *address;
#endif
}

}
}


namespace Saiga {
namespace CUDA{

__device__ inline
double fetch_double(uint2 p){
    return __hiloint2double(p.y, p.x);
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300
//shfl for 64 bit datatypes is already defined in sm_30_intrinsics.h
#else
__device__ inline
double __shfl_down(double var, unsigned int srcLane, int width=32) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return 0;
    return *reinterpret_cast<double*>(&a);
}
#endif

template<typename T, typename ShuffleType = int>
__device__ inline
T shfl(T var, unsigned int srcLane, int width=WARP_SIZE) {
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for(int i = 0 ; i < sizeof(T) / sizeof(ShuffleType) ; ++i){
        a[i] = __shfl(a[i], srcLane, width);
    }
    return var;
}

template<typename T, typename ShuffleType = int>
__device__ inline
T shfl_down(T var, unsigned int srcLane, int width=WARP_SIZE) {
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for(int i = 0 ; i < sizeof(T) / sizeof(ShuffleType) ; ++i){
        a[i] = __shfl_down(a[i], srcLane, width);
    }
    return var;
}

template<typename T, typename ShuffleType = int>
__device__ inline
T shfl_xor(T var, unsigned int srcLane, int width=WARP_SIZE) {
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for(int i = 0 ; i < sizeof(T) / sizeof(ShuffleType) ; ++i){
        a[i] = __shfl_xor(a[i], srcLane, width);
    }
    return var;
}

}
}

namespace Saiga {
namespace CUDA{

template<typename T, unsigned int LOCAL_WARP_SIZE=32, bool RESULT_FOR_ALL_THREADS=false, typename ShuffleType = T>
__device__ inline
T warpReduceSum(T val) {
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE/2; offset > 0; offset /= 2){
//        auto v = RESULT_FOR_ALL_THREADS ? __shfl_xor(val, offset) : __shfl_down(val, offset);
        auto v = RESULT_FOR_ALL_THREADS ? shfl_xor<T,ShuffleType>(val, offset) : shfl_down<T,ShuffleType>(val, offset);
        val = val + v;
    }
    return val;
}

template<typename T, unsigned int LOCAL_WARP_SIZE=32, bool RESULT_FOR_ALL_THREADS=false>
__device__ inline
T warpReduceMax(T val) {
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE/2; offset > 0; offset /= 2){
        auto v = RESULT_FOR_ALL_THREADS ? __shfl_xor(val, offset) : __shfl_down(val, offset);
        val = max(val , v);
    }
    return val;
}




template<typename T, unsigned int BLOCK_SIZE>
__device__ inline
T blockReduceSum(T val, T* shared) {
    int lane = threadIdx.x & (WARP_SIZE-1);
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane==0) shared[wid]=val;

    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum<T,BLOCK_SIZE/WARP_SIZE>(val); //Final reduce within first warp
    return val;
}

template<typename T, unsigned int BLOCK_SIZE>
__device__ inline
T blockReduceAtomicSum(T val, T* shared) {
    int lane = threadIdx.x & (WARP_SIZE-1);

    if(threadIdx.x == 0)
        shared[0] = T(0);

    __syncthreads();

    val = warpReduceSum(val);

    if (lane==0){
        atomicAdd(&shared[0],val);
    }

    __syncthreads();


    if(threadIdx.x == 0)
        val = shared[0];

    return val;
}





template<typename T, unsigned int BLOCK_SIZE>
__device__ inline
T reduceLocalVector(array_view<T> in){
    T sum = T(0);
    unsigned int N = in.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;

    if(sizeof(T) == 4){
        //use vectorized loads for 4 byte types like int/float
        //we could also use the 16 byte load by using int4 below, but
        //this isn't faster than the 8 byte load
        using vector_type = int2;
        const unsigned int elements_per_vector = sizeof(vector_type) / sizeof(T);
//        int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        for(auto i = ti.thread_id; i < N / elements_per_vector; i += ti.grid_size){
            T locals[elements_per_vector];
            vectorArrayCopy<T,vector_type>( in.data() + (i*elements_per_vector) , locals);
#pragma unroll
            for(auto i = 0 ; i < elements_per_vector; ++i)
                sum += locals[i];
        }
        //process remaining elements
        for(auto i = ti.thread_id + N/elements_per_vector * elements_per_vector; i<N; i += ti.grid_size){
            sum += in[i];
        }
    }else{
        for(auto i = ti.thread_id; i < in.size(); i += ti.grid_size){
            sum += in[i];
        }
    }
    return sum;
}



template<typename T, unsigned int BLOCK_SIZE>
__global__
void reduceBlockShared(array_view<T> in, T* out) {
    __shared__ T shared[BLOCK_SIZE/WARP_SIZE];

    T sum = reduceLocalVector<T,BLOCK_SIZE>(in);
    sum = blockReduceSum<T,BLOCK_SIZE>(sum,shared);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

template<typename T, unsigned int BLOCK_SIZE>
__global__
void reduceBlockSharedAtomic(array_view<T> in, T* out) {
    __shared__ T shared;
    T sum = reduceLocalVector<T,BLOCK_SIZE>(in);
    sum = blockReduceAtomicSum<T,BLOCK_SIZE>(sum,&shared);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}


template<typename T, unsigned int BLOCK_SIZE>
__global__
void reduceAtomic(array_view<T> in, T* out) {
    T sum = reduceLocalVector<T,BLOCK_SIZE>(in);
    sum = warpReduceSum(sum);
    int lane = threadIdx.x & (WARP_SIZE-1);
    if(lane == 0)
        atomicAdd(out, sum);
}

}
}

