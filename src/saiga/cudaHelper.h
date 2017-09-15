/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/common.h"
#include "saiga/array_view.h"
#include "saiga/cudaTimer.h"
#include "saiga/imath.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "saiga/assert.h"




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
