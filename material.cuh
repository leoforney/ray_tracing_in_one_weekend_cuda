//
// Created by leo on 3/20/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_MATERIAL_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_MATERIAL_CUH

#include <curand_kernel.h>
#include "rtweekend.h"
#include "color.cuh"

class hit_record;

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const color& a) : albedo(a) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
        auto scatter_direction = rec.normal + random_in_unit_sphere(local_rand_state);

        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
    color albedo;
};

class metal : public material {
public:
    __device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    color albedo;
    float fuzz;
};

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_MATERIAL_CUH
