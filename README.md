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


## Timings

For all meassurements the median time of 50 executions was taken. Host-Device data transfers for example uploading the image for feature detection is not included. The test code can be found in src/test.cu and  is run by default.


### Feature Detection

Sift Parameters:

* doubleScale = false
* octaveLayers = 3
* contrastThresold = 0.04
* edgeThreshold = 10
* sigma = 1.6
* maxDescriptorRadius = 16


| Image Size  |      640x480      | 1280x720 | 1920x1080 | 3000x1688 | 4096x2304 | 
| --- | --- | --- | --- |--- |--- |
| #Features | 990 | 1408 | 1700 | 3382 |6184 |
| GTX 1080 | 1.185 | 1.867 | 2.683 | 4.974 |8.445 |
| GTX 970 | 2.393 | 3.569| 4.639 | 8.513 |14.133 |
| GTX 760 | 3.245 | 5.630 | 8.651 | 17.639 |31.302 |

### Matching

Bruteforce knn search with k=4 

| Match Size   |      990x990      | 1408x1408 | 1700x1700 | 3382x3382 |  6184x6184 | 
| --- | --- | --- | --- |--- |--- |
| GTX 1080 | 0.281 | 0.559 | 0.837 | 3.275 |11.010 |
| GTX 970 | 0.708 | 1.396 | 2.095 | 7.976 |25.953 |
| GTX 760 | 1.161 | 2.311 | 3.472 | 13.231 |44.369 |

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

## Optional Dependencies

 * [Saiga](https://github.com/darglein/saiga) 
 
The important files of saiga are included in src/dep. If you have saiga installed on your system, the installed library will be used.

 * [OpenCV](https://github.com/opencv/opencv) 
 
For running the sample.

## License

The OCS ocde is licensed under the MIT License, but you also have to comply with [OpenCV's license](https://github.com/opencv/opencv_contrib/blob/master/LICENSE). The SIFT algorithm itself is [patented](https://www.google.com/patents/US6711293) and not free for commercial use.

