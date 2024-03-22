//
// Created by leo on 1/28/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH

#include "vec3.cuh"

#include <iostream>

using color = vec3;

__device__ inline float linear_to_gamma(float linear_component) {
    return sqrt(linear_component);
}

void write_color(std::ostream &out, color pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    static const interval intensity(0.000, 0.999);
    out << static_cast<int>(256 * intensity.clamp(r)) << ' '
        << static_cast<int>(256 * intensity.clamp(g)) << ' '
        << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH
