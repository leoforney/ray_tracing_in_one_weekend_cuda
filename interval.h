//
// Created by leo on 3/3/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_INTERVAL_H
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_INTERVAL_H

#include "rtweekend.h"

class interval {
public:
    double min, max;
    interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    interval(double _min, double _max) : min(_min), max(_max) {}

    bool contains(double x) const {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const {
        return min < x && x < max;
    }

    static const interval empty, universe;
};

const static interval empty   (+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_INTERVAL_H
