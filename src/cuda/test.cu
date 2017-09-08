/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "cuda/cudaSift.h"
#include "cuda/matching.h"
#include "cuda/wrapper.h"
#include "saiga/opencv/opencv.h"


void detectedKeypointsTest(){
    std::vector<std::string> imageFiles = {
        "small.jpg",
        "medium.jpg",
        "big.jpg",
        "landscape_small.jpg",
        "landscape.jpg",
    };

	int iterations = 50;

    for(auto str : imageFiles){

        //load image with opencv
        cv::Mat1f img = cv::imread("data/"+str,cv::IMREAD_GRAYSCALE);
        ImageView<float> iv = Saiga::MatToImageView<float>(img);
        Saiga::CUDA::CudaImage<float> cimg(iv);
        cout << "Image " << str << " Size: " << cimg.width << "x" << cimg.height << endl;

        //initialize sift and init memory. Note: this object can be used for multiple
        //images of the same size
        int maxFeatures = 10000;
        SIFTGPU sift(cimg.width,cimg.height,false,-1,maxFeatures,3,0.04,10,1.6);
        sift.initMemory();

        //extract keypoints and descriptors and store them in gpu memory
        thrust::device_vector<SiftPoint> keypoints(maxFeatures);
        thrust::device_vector<float> descriptors(maxFeatures * 128);
        int extractedPoints;
        float time = 34544563456565;
        {
            
			for (int i = 0; i < iterations; ++i) {
				float t;
				{
					Saiga::CUDA::CudaScopedTimer timer(t);
					extractedPoints = sift.compute(cimg, keypoints, descriptors);
				}
				//optimistic minimum timer :)
				time = std::min(t, time);
			}
				
        }

        cout << "Extracted " << extractedPoints << " keypoints in " << time << "ms." << endl;

        //copy to host
        std::vector<SiftPoint> hkeypoints(extractedPoints);
        thrust::copy(keypoints.begin(),keypoints.begin()+extractedPoints,hkeypoints.begin());

        //convert to cvkeypoints
        std::vector<cv::KeyPoint> cvkeypoints;
        SiftWrapper::KeypointsToCV(hkeypoints,cvkeypoints);

        //create debug image
        cv::Mat output;
        img.convertTo(output,CV_8UC1);
        cv::drawKeypoints(output, cvkeypoints, output,cv::Scalar(0,255,0,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

        cv::imwrite("data/"+str+".features.jpg",output);
        cout << endl;

    }
}


void matchTest(){
    std::vector<std::string> imageFiles1 = {
        "small.jpg",
        "medium.jpg",
        "big.jpg",
        "landscape_small.jpg",
        "landscape.jpg",
    };
    std::vector<std::string> imageFiles2 = {
        "small.jpg",
        "medium.jpg",
        "big.jpg",
        "landscape_small.jpg",
        "landscape.jpg",
    };
	int iterations = 50;

    for(int i =0; i < imageFiles1.size() ; ++i){

        //load image with opencv
        cv::Mat1f img1 = cv::imread("data/"+imageFiles1[i],cv::IMREAD_GRAYSCALE);
        cv::Mat1f img2 = cv::imread("data/"+imageFiles2[i],cv::IMREAD_GRAYSCALE);

        Saiga::CUDA::CudaImage<float> cimg1(img1.cols,img1.rows,Saiga::iAlignUp(img1.cols*sizeof(float),256));
        copyImage(Saiga::MatToImageView<float>(img1),cimg1,cudaMemcpyHostToDevice);

        Saiga::CUDA::CudaImage<float> cimg2(img2.cols,img2.rows,Saiga::iAlignUp(img2.cols*sizeof(float),256));
        copyImage(Saiga::MatToImageView<float>(img2),cimg2,cudaMemcpyHostToDevice);
//        Saiga::CUDA::CudaImage<float> cimg1(Saiga::MatToImageView<float>(img1));
//        Saiga::CUDA::CudaImage<float> cimg2(Saiga::MatToImageView<float>(img2));

        int maxFeatures = 10000;
        SIFTGPU sift(cimg1.width,cimg1.height,false,-1,maxFeatures,3,0.04,10,1.6);
        sift.initMemory();

        //extract keypoints and descriptors and store them in gpu memory
        thrust::device_vector<SiftPoint> keypoints1(maxFeatures), keypoints2(maxFeatures);
        thrust::device_vector<float> descriptors1(maxFeatures * 128), descriptors2(maxFeatures * 128);

        int extractedPoints1 = sift.compute(cimg1,keypoints1,descriptors1);
        int extractedPoints2 = sift.compute(cimg2,keypoints2,descriptors2);


        cout << "Match size: " << extractedPoints1 << "x" << extractedPoints2 << endl;

        MatchGPU matcher( std::max(extractedPoints1,extractedPoints2) );
        matcher.initMemory();

        int K = 4;
        thrust::device_vector<float> distances(extractedPoints1 * K);
        thrust::device_vector<int> indices(extractedPoints1 * K);
		float time = 35453426436346;
		{
			for (int i = 0; i < iterations; ++i) {
				float t;
				{
					Saiga::CUDA::CudaScopedTimer timer(t);
					matcher.knnMatch(Saiga::array_view<float>(descriptors1).slice_n(0, extractedPoints1 * 128),
						Saiga::array_view<float>(descriptors2).slice_n(0, extractedPoints2 * 128),
						distances, indices, K
					);
				}
				//optimistic minimum timer :)
				time = std::min(t, time);
			}
        }
		cout << "knnMatch finished in " << time  << "ms." << endl;

        //copy to host
        thrust::host_vector<float> hdistances = distances;
        thrust::host_vector<int> hindices = indices;

        std::vector<cv::DMatch> cvmatches;
        //apply ratio test and convert to cv::DMatch
        for(int i = 0; i < extractedPoints1; ++i){
            float d1 = hdistances[i*K+0];
            float d2 = hdistances[i*K+1];
            if(d1 < 0.7f * d2){
                int id = hindices[i*K+0];
                cv::DMatch m;
                m.distance = d1;
                m.queryIdx = i;
                m.trainIdx = id;
                cvmatches.push_back(m);
            }
        }
        cout << "Number of good matches: " << cvmatches.size() << endl;


        std::vector<SiftPoint> hkeypoints1(extractedPoints1), hkeypoints2(extractedPoints2);
        thrust::copy(keypoints1.begin(),keypoints1.begin()+extractedPoints1,hkeypoints1.begin());
        thrust::copy(keypoints2.begin(),keypoints2.begin()+extractedPoints2,hkeypoints2.begin());
        //convert to cvkeypoints
        std::vector<cv::KeyPoint> cvkeypoints1, cvkeypoints2;
        SiftWrapper::KeypointsToCV(hkeypoints1,cvkeypoints1);
        SiftWrapper::KeypointsToCV(hkeypoints2,cvkeypoints2);

        {
            cv::Mat img1 = cv::imread("data/"+imageFiles1[i]);
            cv::Mat img2 = cv::imread("data/"+imageFiles2[i]);
            //create debug match image
            cv::Mat outImg;
            cv::drawMatches(img1,cvkeypoints1,img2,cvkeypoints2,cvmatches,outImg,cv::Scalar(0,0,255),cv::Scalar(0,255,0),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imwrite("data/matches_"+imageFiles1[i]+"_"+imageFiles2[i]+".jpg",outImg);
        }

        cout << endl;
    }

}

