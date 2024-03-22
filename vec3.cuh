//
// Created by leo on 1/28/24.
//

#ifndef RAY_TRACING_IN_ONE_WEEKEND_CUDA_VEC3_CUH
#define RAY_TRACING_IN_ONE_WEEKEND_CUDA_VEC3_CUH

#include <cmath>
#include <curand_kernel.h>

using std::sqrt;

class vec3 {
public:
    float e[3];

    __host__ __device__ vec3() : e{0, 0, 0} {}

    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ inline float x() const { return e[0]; }

    __host__ __device__ inline float y() const { return e[1]; }

    __host__ __device__ inline float z() const { return e[2]; }
    
    __host__ __device__ inline float r() const { return e[0]; }

    __host__ __device__ inline float g() const { return e[1]; }

    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    __host__ __device__ inline float operator[](int i) const { return e[i]; }

    __host__ __device__ inline float &operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3 &operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(const vec3 &v){
        e[0]  *= v.e[0];
        e[1]  *= v.e[1];
        e[2]  *= v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3 &operator/=(float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ inline float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ inline float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ inline bool near_zero() const {
        auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }
};

using point3 = vec3;

inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
           + u.e[1] * v.e[1]
           + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cost_theta = min(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cost_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif //RAY_TRACING_IN_ONE_WEEKEND_CUDA_VEC3_CUH
