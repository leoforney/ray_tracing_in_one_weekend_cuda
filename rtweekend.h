//
// Created by leo on 3/3/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_RTWEEKEND_H
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <random>
#include <cfloat>

using std::sqrt;

const float infinity = INFINITY;
const float pi = 3.1415926535897932385;
const float moveSpeed = 0.005;

__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

#include "interval.h"
#include "ray.cuh"
#include "vec3.cuh"

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_RTWEEKEND_H
