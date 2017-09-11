/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/tests/test.h"

void detectedKeypointsTest();
void matchTest();


int main(int argc, char *argv[]) {

	Saiga::CUDA::initCUDA();
	//Saiga::CUDA::testCuda();
	//Saiga::CUDA::testThrust();

    detectedKeypointsTest();
//    matchTest();
    return 0;
}
