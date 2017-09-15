/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>

namespace cudasift {
void detectedKeypointsTest();
void matchTest();
}

int main(int argc, char *argv[]) {
    cudasift::detectedKeypointsTest();
    cudasift::matchTest();
    return 0;
}
