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

class dielectric : public material {
public:
    __device__ dielectric(float index_of_reflection) : ir(index_of_reflection) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());

        float cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }
    float ir;

    __device__ static float reflectance(float cosine, float ref_idx) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine),5);
    }
};

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_MATERIAL_CUH
