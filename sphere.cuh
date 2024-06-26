//
// Created by leo on 3/3/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_SPHERE_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_SPHERE_CUH

#include <utility>

#include "hittable.cuh"
#include "vec3.cuh"

class sphere: public hittable {
public:
    __device__ sphere() {}
    __device__ sphere(point3 _center, float _radius, material *_material)
    : center(_center), radius(_radius), mat(_material) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = r.origin() - center;
        auto a = r.direction().length_squared();
        auto half_b = dot(oc, r.direction());
        auto c = oc.length_squared() - radius * radius;
        auto discriminant = half_b * half_b - a * c;

        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        auto root = (-half_b - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

    material *mat;
    point3 center;
    float radius;
};

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_SPHERE_CUH
