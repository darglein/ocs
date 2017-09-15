/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/common.h"


#include <algorithm>

namespace Saiga {

template<typename T>
class Statistics{
public:
    Statistics(const std::vector<T>& data);

    int numValues;
    T min;
    T max;
    T median;
    T mean, sum;
    T variance, sdev;
};

template<typename T>
Statistics<T>::Statistics(const std::vector<T>& _data)
{
    std::vector<T> data = _data;
    numValues = data.size();

    if(numValues  == 0)
        return;

    std::sort(data.begin(),data.end());

    min = data.front();
    max = data.back();
    median = data[data.size()/2];

    sum = 0;
    for(auto &d : data)
        sum += d;
    mean = sum / numValues;

    variance = 0;
    for(auto d : data){
        d = d - mean;
        variance += d * d;
    }
    variance /= numValues;

    sdev = std::sqrt(variance);
}

template<typename T>
std::ostream &operator<<(std::ostream &stream, const Statistics<T> &object)
{
    stream << "Num         = [" << object.numValues << "]" << std::endl <<
              "Min,Max     = [" << object.min << "," << object.max << "]" << std::endl <<
              "Mean,Median = [" << object.mean << "," << object.median << "]" << std::endl <<
              "sdev,var    = [" << object.sdev << "," << object.variance << "]";
    return stream;
}

}

namespace Saiga {
namespace CUDA {


class  CudaScopedTimer{
    float& time;
    cudaEvent_t start, stop;
public:
    CudaScopedTimer(float& time);
    ~CudaScopedTimer();
};



class  CudaScopedTimerPrint {
    std::string name;
    cudaEvent_t start, stop;
public:
    CudaScopedTimerPrint(const std::string &name);
    ~CudaScopedTimerPrint();
};


}
}



namespace Saiga {


template<typename TimerType, typename F, typename ... Ts>
inline void measureFunction(const std::string& name, int its, F f, Ts&... args)
{
    std::vector<float> timings(its);
    for(int i = 0; i < its; ++i){
        float time;
        {
            TimerType tim(time);
            f(args...);
        }
        timings[i] = time;
    }
    std::cout << "> Measured execution time of function " << name << " in ms." << std::endl;
    std::cout << Statistics<float>(timings) << std::endl;
}



template<typename TimerType, typename F>
inline
Statistics<float> measureObject(int its, F f)
{
    std::vector<float> timings(its);
    for(int i = 0; i < its; ++i){
        float time;
        {
            TimerType tim(time);
            f();
        }
        timings[i] = time;
    };
   return Statistics<float>(timings);
}

template<typename TimerType, typename F>
inline void measureObject(const std::string& name, int its, F f)
{
    auto st = measureObject<TimerType>(its,f);
    std::cout << "> Measured execution time of function " << name << " in ms." << std::endl;
    std::cout << st << std::endl;
}


}
