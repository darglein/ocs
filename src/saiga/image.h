/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <thrust/device_vector.h>


#include "saiga/common.h"
#include <algorithm>

#if defined(SAIGA_USE_CUDA)
#include <vector_types.h>
#else
#if !defined (__VECTOR_TYPES_H__)
struct uchar3
{
    unsigned char x, y, z;
};

struct GLM_ALIGN(4) uchar4
{
    unsigned char x, y, z, w;
};
#endif
#endif

namespace Saiga {


template<typename T>
struct ImageView{
    union{
        int width;
        int cols;
    };
    union{
        int height;
        int rows;
    };
    //    int width, height;
    //    int pitch; //important: the pitch is not in bytes!!!
    int pitchBytes;

    union{
        void* data;
        uint8_t* data8;
    };



    HD inline
    ImageView(){
        static_assert(sizeof(ImageView<T>) == 24, "ImageView size wrong!");
    }

    HD inline
    ImageView(int h, int w , int p, void* data)
        : width(w),height(h),pitchBytes(p),data(data) {}

    HD inline
    ImageView(int h, int w, void* data)
        : width(w),height(h),pitchBytes(w*sizeof(T)),data(data) {}

    //size in bytes
    HD inline
    int size(){
        return height * pitchBytes;
    }

    //a view to a sub image
    HD inline
    ImageView<T> subImageView7(int startY, int startX, int h, int w){
#ifdef ON_DEVICE
#else
        SAIGA_ASSERT(startX >= 0 && startX < width);
        SAIGA_ASSERT(startY >= 0 && startY < height);
        SAIGA_ASSERT(startX + w <= width);
        SAIGA_ASSERT(startY + h <= height);
#endif
        ImageView<T> iv(w,h,pitchBytes,&atIVxxx(startY,startX));
        return iv;
    }

#if 0
    HD inline
    T& operator()(int x, int y){
#ifdef ON_DEVICE
#else
        SAIGA_ASSERT(inImage(x,y));
#endif
        //        return data[y * pitch + x];
//        uint8_t* data8 = reinterpret_cast<uint8_t*>(data);
//        data8 += y * pitchBytes + x * sizeof(T);
        auto ptr = data8 + y * pitchBytes + x * sizeof(T);
        return reinterpret_cast<T*>(ptr)[0];
    }
#endif

    HD inline
    T& atIVxxx(int y, int x){
        return rowPtr(y)[x];
    }


    HD inline
    T* rowPtr(int y){
        auto ptr = data8 + y * pitchBytes;
        return reinterpret_cast<T*>(ptr);
    }





    //bilinear interpolated pixel with clamp to edge boundary
    HD inline
    T inter7(float sy, float sx){

        int x0 = iFloor(sx);
        int y0 = iFloor(sy);

        //interpolation weights
        float ax = sx - x0;
        float ay = sy - y0;

        if(x0 < 0){ x0=0;ax=0;};
        if ( x0>=width ) {x0=width-1;ax=0;}
        if ( y0<0 ) {y0=0;ay=0;}
        if ( y0>=height ) {y0=height-1;ay=0;}


#ifdef ON_DEVICE
        int x1 = min(x0 + 1, width - 1);
        int y1 = min(y0 + 1, height - 1);
#else
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);
#endif



        T res = ((*this)(x0,y0) * (1.0f - ax) + (*this)(x1,y0) * (ax)) * (1.0f - ay) +
                ((*this)(x0,y1) * (1.0f - ax) + (*this)(x1,y1) * (ax)) * (ay);
        return res;
    }

    HD inline
    bool inImage7(int y, int x){
        return x >= 0 && x < width && y >=0 && y < height;
    }

    //minimum distance of the pixel to all edges
    HD inline
    int distanceFromEdge7(int y, int x){
        int x0 = x;
        int x1 = width - 1 - x;
        int y0 = y;
        int y1 = height - 1 - y;
#ifdef ON_DEVICE
        return min(x0,min(x1,min(y0,y1)));
#else
        return std::min(x0,std::min(x1,std::min(y0,y1)));
#endif
    }

    template<typename AT>
    HD inline
    bool inImage7(AT y, AT x){
        return x >= 0 && x <= AT(width-1) && y >=0 && y <= AT(height-1);
    }

    template<typename AT>
    HD inline
    void multWithScalar(AT a){
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                (*this)(x,y) *= a;
            }
        }
    }

    HD inline
    void clampToEdge7(int& y, int& x){
#ifdef ON_DEVICE
        x = min(max(0,x),width-1);
        y = min(max(0,y),height-1);
#else
        x = std::min(std::max(0,x),width-1);
        y = std::min(std::max(0,y),height-1);
#endif
    }

    HD inline
    T clampedRead7(int y, int x){
        clampToEdge7(y,x);
        return atIVxxx(y,x);
    }


    HD inline
    T borderRead7(int y, int x, const T& borderValue){
        return inImage7(y,x) ? atIVxxx(y,x) : borderValue;
    }

    //write only if the point is in the image
    HD inline
    void clampedWrite7(int y, int x, const T& v){
        if(inImage7(y,x))
            atIVxxx(y,x) = v;
    }
};

//multiple images that are stored in memory consecutively
template<typename T>
struct ImageArrayView{
    ImageView<T> imgStart;
    int n;

    ImageArrayView(){}
    ImageArrayView(ImageView<T> _imgStart, int _n) : imgStart(_imgStart), n(_n) {}

    HD inline
    ImageView<T> at(int i){
        ImageView<T> res = imgStart;
        res.data =  imgStart.data8 + imgStart.size() * i;
        return res;
    }

    HD inline
    ImageView<T> operator[](int i){ return at(i); }

//    HD inline
//    T& operator()(int x, int y, int z){
//        auto ptr = imgStart.data8 + z * imgStart.size() + y * imgStart.pitchBytes + x * sizeof(T);
//        return reinterpret_cast<T*>(ptr)[0];
//    }

    HD inline
    T& atIARVxxx(int z, int y, int x){
        auto ptr = imgStart.data8 + z * imgStart.size() + y * imgStart.pitchBytes + x * sizeof(T);
        return reinterpret_cast<T*>(ptr)[0];
    }
};

}

namespace Saiga {
namespace CUDA {



template<typename T>
void copyImage(ImageView<T> imgSrc, ImageView<T> imgDst, enum cudaMemcpyKind kind){
    CHECK_CUDA_ERROR(cudaMemcpy2D(imgDst.data,imgDst.pitchBytes,imgSrc.data,imgSrc.pitchBytes,imgSrc.width*sizeof(T),imgSrc.height,kind));
}



//with these two functions we are able to use CudaImage from cpp files.
 void resizeDeviceVector(thrust::device_vector<uint8_t>& v, int size);
 void copyDeviceVector(const thrust::device_vector<uint8_t>& src, thrust::device_vector<uint8_t>& dst);

//supported types:
//float, uchar3, uchar4
template<typename T>
struct CudaImage : public ImageView<T>{
    thrust::device_vector<uint8_t> v;

    CudaImage(){}

    CudaImage(int h, int w , int p)
        : ImageView<T>(h,w,p,0) {
        create();
    }

    CudaImage(int h, int w)
        : ImageView<T>(h,w,0) {
        create();
    }


    CudaImage(ImageView<T> h_img) {
        upload(h_img);
    }

    //upload a host imageview to the device
    inline
    void upload(ImageView<T> h_img){
        this->ImageView<T>::operator=(h_img);
        create();
        copyImage(h_img,*this,cudaMemcpyHostToDevice);
    }

    //download a host imageview from the device
    inline
    void download(ImageView<T> h_img){
        copyImage(*this,h_img,cudaMemcpyDeviceToHost);
    }

    inline void create(){
        resizeDeviceVector(v,this->size());
        this->data = thrust::raw_pointer_cast(v.data());
    }

    inline void create(int h, int w){
        create(h,w,w*sizeof(T));
    }


    inline void create(int h, int w , int p){
        this->width = w;
        this->height = h;
        this->pitchBytes = p;
        create();
    }

    //copy and swap idiom
    //http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    CudaImage(CudaImage const& other) : ImageView<T>(other){
		copyDeviceVector(other.v, v);
        this->data = thrust::raw_pointer_cast(v.data());
    }

    CudaImage& operator=(CudaImage other){
        swap(*this, other);
        return *this;
    }

    template<typename T2>
    friend void swap(CudaImage<T2>& first, CudaImage<T2>& second);
};

template<typename T>
inline
void swap(CudaImage<T> &first, CudaImage<T> &second)
{
    using std::swap;
    first.v.swap(second.v);
    swap(first.width,second.width);
    swap(first.height,second.height);
    swap(first.pitchBytes,second.pitchBytes);
    swap(first.data,second.data);
}

}
}
