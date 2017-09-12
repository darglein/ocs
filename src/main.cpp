/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include "saiga/cuda/cudaHelper.h"

void detectedKeypointsTest();
void matchTest();


int main(int argc, char *argv[]) {

	Saiga::CUDA::initCUDA();

    detectedKeypointsTest();
    matchTest();
    return 0;
}
