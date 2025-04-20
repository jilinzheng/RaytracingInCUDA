#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"
#include "color.h"


enum class MaterialType {
    LAMBERTIAN,
    METAL
};

// generic material struct
struct material {
    MaterialType type;
    color albedo;
    float fuzz = 0.0f;

    __host__ __device__ material(MaterialType type, color albedo)
        : type(type), albedo(albedo) {}
    __host__ __device__ material(MaterialType t, color a, float f)
        : type(t), albedo(a), fuzz(fuzz) {}
};


__device__ bool lambertian_scatter(const ray& r_in, const hit_record& rec,
    color& attenuation, ray& scattered, curandState *thread_rand_state) {
    // choose lambertian diffuse reflectance to always scatter
    vec3 scatter_direction = rec.normal
                            + random_unit_vector(thread_rand_state);
    // catch degenerate scatter direction
    if (scatter_direction.near_zero()) scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction);
    attenuation = rec.mat->albedo;
    return true;
}

__device__ bool metal_scatter(const ray& r_in, const hit_record& rec,
    color& attenuation, ray& scattered, curandState *thread_rand_state) {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected)+(rec.mat->fuzz*random_unit_vector(thread_rand_state));
    scattered = ray(rec.p, reflected);
    attenuation = rec.mat->albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}


#endif
