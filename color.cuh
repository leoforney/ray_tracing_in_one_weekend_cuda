//
// Created by leo on 1/28/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH

#include "vec3.cuh"

#include <iostream>

using color = vec3;

__host__ __device__ inline float linear_to_gamma(float linear_component) {
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

Uint32 convertColorToUint32(const vec3& color) {
    static const interval intensity(0.000, 0.999);
    Uint8 r = static_cast<int>(256 * intensity.clamp(color.r()));
    Uint8 g = static_cast<int>(256 * intensity.clamp(color.g()));
    Uint8 b = static_cast<int>(256 * intensity.clamp(color.b()));
    return (0xFF << 24) | (r << 16) | (g << 8) | b;
}

void updateTextureFromFrameBuffer(SDL_Texture *texture, color *fb, const unsigned int numXPixels, const unsigned int numYPixels) {
    Uint32* pixels = new Uint32[numXPixels * numYPixels];
    int index = 0; // Index for the 'pixels' array
    for (int j = numYPixels - 1; j >= 0; j--) {
        for (int i = 0; i < numXPixels; ++i) {
            pixels[index++] = convertColorToUint32(fb[j * numXPixels + i]);
        }
    }

    SDL_UpdateTexture(texture, NULL, pixels, numXPixels * sizeof(Uint32));
    delete[] pixels;
}

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_COLOR_CUH
