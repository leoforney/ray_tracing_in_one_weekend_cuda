//
// Created by leo on 1/28/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH

#include "vec3.cuh"

#include <iostream>

using color = vec3;

void write_color(std::ostream &out, color pixel_color) {
    out << static_cast<int>(255.999 * pixel_color.x()) << ' '
        << static_cast<int>(255.999 * pixel_color.y()) << ' '
        << static_cast<int>(255.999 * pixel_color.z()) << '\n';
}

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH
