#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"
#include "color.h"
#include "rtweekend.h"


// material type identifier
enum class MaterialType {
    LAMBERTIAN,
    METAL,
    DIELETRIC
};

// generic material struct
struct material {
    MaterialType type;
    color albedo;
    double fuzz;
    double refraction_index;

    __host__ __device__ material() {}
    // lambertian constructor
    __host__ __device__ material(MaterialType type, color albedo)
        : type(type), albedo(albedo) {}
    // metal constructor
    __host__ __device__ material(MaterialType type, color albedo, double fuzz)
        : type(type), albedo(albedo), fuzz(fuzz < 1.0 ? fuzz : 1.0) {}
    // dieletric constructor
    __host__ __device__ material(MaterialType type, double refraction_index)
        : type(type), refraction_index(refraction_index) {}
};


// material-specific scattering functions
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
    reflected = unit_vector(reflected)
                + (rec.mat->fuzz * random_unit_vector(thread_rand_state));
    scattered = ray(rec.p, reflected);
    attenuation = rec.mat->albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

// Schlick's approximation for reflectance
__device__ inline double reflectance(double cosine, double refraction_index) {
    double r0 = (1-refraction_index)/(1+refraction_index);
    r0 = r0*r0;
    return r0 + (1-r0)*powf((1-cosine),5);
}

__device__ bool dieletric_scatter(const ray& r_in, const hit_record& rec,
    color& attenuation, ray& scattered, curandState *thread_rand_state) {
    attenuation = color(1.0,1.0,1.0);
    double refraction_index = rec.mat->refraction_index;

    double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

    vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    vec3 direction;

    // Schlick's approximation for reflectance
    if (cannot_refract || reflectance(cos_theta, ri) > device_random_double(thread_rand_state))
        direction = reflect(unit_direction, rec.normal);
    else direction = refract(unit_direction, rec.normal, ri);

    scattered = ray(rec.p, direction);
    return true;
}

#endif
