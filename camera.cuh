//
// Created by leo on 3/18/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_CAMERA_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_CAMERA_CUH

#include "rtweekend.h"

#include "color.cuh"
#include "hittable.cuh"
#include "material.cuh"

#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <cfloat>

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) {
        origin = lookfrom;
        target = lookat;
        view_up = vup;
        aspect_ratio = aspect;
        fov = vfov;
        float dist_to_focus = (lookfrom-lookat).length();
        float aperture = 2.0;
        update_params();
    }
    __device__ ray get_ray(float u, float v, curandState *local_rand_state) {
        return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    }

    vec3 origin;
    vec3 target;
    vec3 view_up;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 u, v, w;
    vec3 vertical;

    float fov;
    float aspect_ratio;
    float lens_radius;

    __device__ void update_params() {
        float theta = degrees_to_radians(fov);
        float half_height = tan(theta / 2);
        float half_width = aspect_ratio * half_height;
        w = unit_vector(origin - target);
        u = unit_vector(cross(view_up, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * u -half_height * v - w;
        horizontal = 2.0f * half_width * u;
        vertical = 2.0f * half_height * v;
    }

    __device__ static vec3 ray_color(const ray& r, hittable **world, curandState *local_rand_state) {
        ray cur_ray = r;
        vec3 cur_attenuation = vec3(1.0,1.0,1.0);
        for(int i = 0; i < 50; i++) {
            hit_record rec;
            if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec)) {
                ray scattered;
                vec3 attenuation;
                if(rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0.0,0.0,0.0);
                }
            }
            else {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                float t = 0.5f*(unit_direction.y() + 1.0f);
                vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
            }
        }
        return vec3(0.0,0.0,0.0);
    }
};

__global__ static void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ static void render(vec3 *fb, int max_x, int max_y, int numRays, camera **cam, hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color col(0,0,0);

    for(int s = 0; s < numRays; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += camera::ray_color(r, world, &local_rand_state);
    }

    fb[pixel_index] = col;

    rand_state[pixel_index] = local_rand_state;
}

__global__ static void accumulate_samples(color *fb, int max_x, int max_y, int numRays) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;

    color col = fb[pixel_index];

    col /= float(numRays);
    col[0] = linear_to_gamma(col[0]);
    col[1] = linear_to_gamma(col[1]);
    col[2] = linear_to_gamma(col[2]);
    fb[pixel_index] = col;
}

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_CAMERA_CUH
