#ifndef VEC3_H
#define VEC3_H

#include "curand_kernel.h"


class vec3 {
  public:
    double e[3];

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    __host__ __device__ double x() const { return e[0]; }
    __host__ __device__ double y() const { return e[1]; }
    __host__ __device__ double z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(double t) {
        return *this *= 1/t;
    }

    __host__ __device__ double length() const {
        return sqrtf(length_squared());
    }

    __host__ __device__ double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __device__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        double s = 1e-6f;
        return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
    }

    static vec3 random() {
        return vec3(random_double(), random_double(), random_double());
    }

    static vec3 random(double min, double max) {
        return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// vector utility functions

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_disk(curandState *rand_state) {
    while (true) {
        vec3 p = vec3(device_random_double(-1,1,rand_state), device_random_double(-1,1,rand_state), 0);
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState *rand_state) {
    while (true) {
        double x = curand_uniform(rand_state) * 2.0f - 1.0f; // range [-1, 1)
        double y = curand_uniform(rand_state) * 2.0f - 1.0f; // range [-1, 1)
        double z = curand_uniform(rand_state) * 2.0f - 1.0f; // range [-1, 1)
        vec3 p(x, y, z);
        double lensq = dot(p, p);
        if (1e-8f < lensq && lensq <= 1.0f) // using a smaller epsilon for double
            return p / sqrtf(lensq);        // use sqrtf for double
    }
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    double cos_theta = fminf(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


#endif
