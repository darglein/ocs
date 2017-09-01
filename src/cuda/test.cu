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
        "landscape.jpg",
    };

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
        float time;
        {
            Saiga::CUDA::CudaScopedTimer timer(time);
            extractedPoints = sift.compute(cimg,keypoints,descriptors);
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
        "medium.jpg",
    };
    std::vector<std::string> imageFiles2 = {
        "medium.jpg",
    };

    for(int i =0; i < imageFiles1.size() ; ++i){

        //load image with opencv
        cv::Mat1f img1 = cv::imread("data/"+imageFiles1[i],cv::IMREAD_GRAYSCALE);
        cv::Mat1f img2 = cv::imread("data/"+imageFiles2[i],cv::IMREAD_GRAYSCALE);

        Saiga::CUDA::CudaImage<float> cimg1(Saiga::MatToImageView<float>(img1));
        Saiga::CUDA::CudaImage<float> cimg2(Saiga::MatToImageView<float>(img2));

        int maxFeatures = 10000;
        SIFTGPU sift(cimg1.width,cimg1.height,false,-1,maxFeatures,3,0.04,10,1.6);
        sift.initMemory();

        //extract keypoints and descriptors and store them in gpu memory
        thrust::device_vector<SiftPoint> keypoints1(maxFeatures), keypoints2(maxFeatures);
        thrust::device_vector<float> descriptors1(maxFeatures * 128), descriptors2(maxFeatures * 128);

        int extractedPoints1 = sift.compute(cimg1,keypoints1,descriptors1);
        int extractedPoints2 = sift.compute(cimg2,keypoints2,descriptors2);



        MatchGPU matcher( std::max(extractedPoints1,extractedPoints2) );
        matcher.initMemory();

        int K = 4;
        thrust::device_vector<float> distances(extractedPoints1 * K);
        thrust::device_vector<int> indices(extractedPoints1 * K);
        {
            Saiga::CUDA::CudaScopedTimerPrint timer("knnmatch");
            matcher.knnMatch( Saiga::array_view<float>(descriptors1).slice_n(0,extractedPoints1*128),
                              Saiga::array_view<float>(descriptors2).slice_n(0,extractedPoints2*128),
                              distances,indices, K
                              );
        }

        //copy to host
        thrust::host_vector<float> hdistances = distances;
        thrust::host_vector<int> hindices = indices;

        std::vector<cv::DMatch> cvmatches;
        //apply ratio test and convert to cv::DMatch
        for(int i = 0; i < extractedPoints1; ++i){
            float d1 = hdistances[i*K+0];
            float d2 = hdistances[i*K+1];
            if(d1 < 0.8f * d2){
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
            cv::drawMatches(img1,cvkeypoints1,img2,cvkeypoints2,cvmatches,outImg);
            cv::imwrite("data/matches.jpg",outImg);
        }

    }

}

