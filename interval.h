//
// Created by leo on 3/3/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_INTERVAL_H
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_INTERVAL_H

#include "rtweekend.h"

class interval {
public:
    float min, max;
    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __host__ __device__ interval(float _min, float _max) : min(_min), max(_max) {}

    __host__ __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    __host__ __device__ float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
};

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_INTERVAL_H
