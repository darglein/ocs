# Open CUDA Sift (OCS)

OCS is an open source GPU implemenation of SIFT feature detection and matching. The CUDA code is a direct port of [OpenCV's SIFT implementation](https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/sift.cpp).


## Differences to OpenCV

 * Added an option to disable subpixel interpolation
 * Added an option to limit the maximum sample radius used in orientation assignment and descriptor computation
 * Output descriptor element range is [-1,1] instead of [-128,127]
 * Removed the "char rounding" at the end of descriptor computation
 * Added an option to limit the maximum number of octave layers
 * Added an option to upscale the image before feature detection
 * Use a fixed gaussian blur kernel size of 9x9


## Timings - Feature Detection (todo)

The following SIFT settings were used for the tests:
doubleScale = false
octaveLayers = 3
contrastThresold = 0.04
edgeThreshold = 10
sigma = 1.6
maxDescriptorRadius = 16


| Image Size   |      640x480      | 1280x720 | 1920x1080 | 4096x2304 | 
|----------|:-------------:|------:|
| GTX 970 |  12.01 | 51.10 | 12.01 | 51.10 |
| GTX 980 Ti |  12.01 | 51.10 | 12.01 | 51.10 |
| GTX 1080 | 12.01 | 51.10 | 12.01 | 51.10 |


## Timings - Matching (todo)

Time in (ms) for the knn search with k=4.


## Dependencies

 * [CUDA 8](https://developer.nvidia.com/cuda-downloads)
 * [Saiga](https://github.com/darglein/saiga) for some basic CUDA image processing
 * [OpenCV](https://github.com/opencv/opencv) only for the samples.

## TODO

 * Remove Saiga dependency by copying the required files.
 * Increase keypoint detection time

