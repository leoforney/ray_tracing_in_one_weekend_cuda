//
// Created by leo on 1/28/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_RAY_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_RAY_CUH

#include "vec3.cuh"

class ray {
public:
    ray() {}

    ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}

    point3 origin() const { return orig; }
    vec3 direction() const { return dir; }

    point3 at(double t) const {
        return orig + t*dir;
    }

private:
    point3 orig;
    vec3 dir;
};

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_RAY_CUH
