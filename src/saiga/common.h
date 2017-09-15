/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once



//remove all CUDA_SYNC_CHECK_ERROR and CUDA_ASSERTS
//for gcc add cppflag: -DCUDA_NDEBUG
#if !defined(CUDA_NDEBUG) 
#if !defined(CUDA_DEBUG)
#define CUDA_DEBUG
#endif
#else
#undef CUDA_DEBUG
#endif


#ifdef __CUDACC__
#	define HD __host__ __device__
#	define IS_CUDA
#	if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#		define ON_DEVICE
#	endif
#else
#	define HD
#   if !defined(__launch_bounds__)
#       define __launch_bounds__
#   endif
#endif


// constants defined as functions, because cuda device code
// can access constexpr functions but not constexpr globals
//HD  inline float PI() { return 3.1415926535897932f; }
//HD  inline float TWOPI() { return 2 * PI(); }
//HD  inline float INV_PI() { return 1.f / PI(); }
//HD  inline float INV_TWOPI() { return 1.f / TWOPI(); }


#define WARP_SIZE 32

#define L1_CACHE_LINE_SIZE 128
#define L2_CACHE_LINE_SIZE 32

#include "saiga/array_view.h"
#include "saiga/cudaTimer.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>




# define CHECK_CUDA_ERROR(cudaFunction) {							\
  cudaError_t  cudaErrorCode = cudaFunction;                                                       \
  ((cudaErrorCode == cudaSuccess)								\
   ? static_cast<void>(0)						\
   : Saiga::saiga_assert_fail (#cudaFunction " == cudaSuccess", __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION,cudaGetErrorString(cudaErrorCode))); \
}

#if defined(CUDA_DEBUG)
# define CUDA_SYNC_CHECK_ERROR() { CHECK_CUDA_ERROR(cudaDeviceSynchronize()); }
#else
# define CUDA_SYNC_CHECK_ERROR()		( static_cast<void>(0))
#endif



//similar to unix assert.h implementation

namespace Saiga {
 extern void saiga_assert_fail (const char *__assertion, const char *__file,
               unsigned int __line, const char *__function, const char *__message);
//      throw __attribute__ ((__noreturn__));
}

# if defined WIN32
#   define SAIGA_ASSERT_FUNCTION	__FUNCSIG__
# else
#include <features.h>
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define SAIGA_ASSERT_FUNCTION	__PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define SAIGA_ASSERT_FUNCTION	__func__
#  else
#   define SAIGA_ASSERT_FUNCTION	((const char *) 0)
#  endif
# endif
# endif



#if defined(SAIGA_ASSERTS)

# define SAIGA_ASSERT_MSG(expr,msg)							\
  ((expr)								\
   ? static_cast<void>(0)						\
   : Saiga::saiga_assert_fail (#expr, __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION,msg))

#else

//# define SAIGA_ASSERT_MSG(expr,msg)		(static_cast<void>(0))

//this is a trick so that no unused variable warnings are generated if a variable
//is only used in an assert
# define SAIGA_ASSERT_MSG(expr,msg)         \
   if(false) static_cast<void>(expr)


#endif

#define SAIGA_ASSERT_NOMSG(expr) SAIGA_ASSERT_MSG(expr,"")


//With this trick SAIGA_ASSERT is overloaded for 1 and 2 arguments. (With and without message)
#define GET_SAIGA_ASSERT_MACRO(_1,_2,NAME,...) NAME
#define SAIGA_ASSERT(...) GET_SAIGA_ASSERT_MACRO(__VA_ARGS__, SAIGA_ASSERT_MSG, SAIGA_ASSERT_NOMSG, 0)(__VA_ARGS__)


//#undef assert



namespace Saiga {

//returns the smallest x number with: x * b >= a
HD constexpr inline
int iDivUp(int a, int b) { return (a + b - 1) / b; }

HD constexpr inline
int iDivDown(int a, int b) { return a / b; }

//finds the smallest number that is bigger or equal than a and divisible by b
HD constexpr inline
int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }

//finds the largest number that is smaller or equal than a and divisible by b
HD constexpr inline
int iAlignDown(int a, int b) {return a - a % b; }

HD inline
int iFloor(float value){
    int i = (int)value;
    return i - (i > value);
}

HD inline
int iCeil(float value){
    int i = (int)value;
    return i + (i < value);
}

HD inline
int iRound(float value){
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}


}
