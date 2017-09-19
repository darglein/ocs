/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <saiga/cuda/tests/test.h>
#include <saiga/cuda/cudaHelper.h>

namespace cudasift {
void detectedKeypointsTest();
void matchTest();
}

int main(int argc, char *argv[]) {
    Saiga::CUDA::initCUDA();

    cudasift::detectedKeypointsTest();
    cudasift::matchTest();

    Saiga::CUDA::destroyCUDA();
    return 0;
}
