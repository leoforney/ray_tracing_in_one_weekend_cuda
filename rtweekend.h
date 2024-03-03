//
// Created by leo on 3/3/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_RTWEEKEND_H
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

#include "interval.h"
#include "ray.cuh"
#include "vec3.cuh"

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_RTWEEKEND_H
