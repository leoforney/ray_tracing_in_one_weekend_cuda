//
// Created by leo on 3/3/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_HITTABLE_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_HITTABLE_CUH

#include "rtweekend.h"

class material;

#include "ray.cuh"

class hit_record {
public:
    point3 p;
    vec3 normal;
    float t;
    bool front_face;
    material *mat;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_HITTABLE_CUH
