# Open CUDA Sift (OCS)

OCS is an open source GPU implemenation of SIFT feature detection and matching. The CUDA code is a direct port of [OpenCV's SIFT implementation](https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/sift.cpp).

<img src="data/features_ref.jpg" width="600"/> 
<img src="data/matches_ref.jpg" /> 

## Differences to OpenCV

 * Added an option to disable subpixel interpolation
 * Added an option to limit the maximum sample radius used in orientation assignment and descriptor computation
 * Output descriptor element range is [-1,1] instead of [-128,127]
 * Removed the "char rounding" at the end of descriptor computation
 * Added an option to limit the maximum number of octave layers
 * Added an option to upscale the image before feature detection
 * Use a fixed gaussian blur kernel size of 9x9


## Timings - Feature Detection

The following SIFT settings were used for the tests:

* doubleScale = false
* octaveLayers = 3
* contrastThresold = 0.04
* edgeThreshold = 10
* sigma = 1.6
* maxDescriptorRadius = 16

All timings are given in milliseconds (ms).

| Image Size  |      640x480      | 1280x720 | 1920x1080 | 3000x1688 | 4096x2304 | 
| --- | --- | --- | --- |--- |--- |
| #Features | 990 | 1408 | 1700 | 3382 |6184 |
| GTX 1080 | - | - | - | - |- |
| GTX 970 | 2.393 | 3.569| 4.639 | 8.513 |14.133 |

## Timings - Matching

Time in (ms) for a knn search with k=4. The match size gives the number of keypoints/descriptors for each image. In the largest test (6184x6184) a total number of 38241856 Sift descriptor comparisons were performed.

| Match Size   |      990x990      | 1408x1408 | 1700x1700 | 3382x3382 |  6184x6184 | 
| --- | --- | --- | --- |--- |--- |
| GTX 1080 | 0.327 | 0.655 | 0.979 | 3.432 |10.947 |
| GTX 970 | 0.708 | 1.396 | 2.095 | 7.976 |25.953 |

## Usage

```c++
//load image and copy it to the device
cv::Mat1f img1 = cv::imread("data/landscape.jpg",cv::IMREAD_GRAYSCALE);
Saiga::CUDA::CudaImage<float> d_img(img.cols,img.rows,Saiga::iAlignUp(img.cols*sizeof(float),256));
copyImage(Saiga::MatToImageView<float>(img),d_img,cudaMemcpyHostToDevice);

//initialize sift and init memory. 
int maxFeatures = 10000;
SIFTGPU sift(d_img.width,d_img.height,false,-1,maxFeatures,3,0.04,10,1.6);
sift.initMemory();

//extract keypoints and descriptors on the gpu
thrust::device_vector<SiftPoint> keypoints(maxFeatures);
thrust::device_vector<float> descriptors(maxFeatures * 128);
int extractedPoints = sift.compute(d_img, keypoints, descriptors)
```

## Dependencies

 * [CUDA 8](https://developer.nvidia.com/cuda-downloads)
 * [Saiga](https://github.com/darglein/saiga) for some basic CUDA image processing
 * [OpenCV](https://github.com/opencv/opencv) only for the samples.

## TODO

 * Remove Saiga dependency by copying the required files.

## License

The OCS ocde is licensed under the MIT License, but you also have to comply with [OpenCV's license](https://github.com/opencv/opencv_contrib/blob/master/LICENSE). The SIFT algorithm itself is [patented](https://www.google.com/patents/US6711293) and not free for commercial use.

