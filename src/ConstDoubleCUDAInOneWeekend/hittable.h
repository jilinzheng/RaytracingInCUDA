#ifndef HITTABLE_H
#define HITTABLE_H

#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "color.h"

#define SCENE_1_NUM_SPHERES (1 + 22 * 22 + 3)


// base struct for hit records (no methods)
struct hit_record {
    point3 p;
    vec3 normal;
    double t;
    bool front_face;
    int mat_idx;
};

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
        : type(type), albedo(albedo), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}
    // dieletric constructor
    __host__ __device__ material(MaterialType type, double refraction_index)
        : type(type), refraction_index(refraction_index) {}
};

// sphere definition (no inheritance)
struct sphere {
    point3 center;
    double radius;
    int mat_idx;

    __host__ __device__ sphere() {}
    __host__ __device__ sphere(point3 cen, double r, int mat_idx)
        : center(cen), radius(r), mat_idx(mat_idx) {}
};

// world consists of sphere and any other objects in the future
// analogous to hittable_list - chapter 6.5
struct world {
    int num_spheres;
    // add other object types as needed

    __host__ __device__ world() {}
    __host__ __device__ world(int ns) : num_spheres(ns) {}
};


extern __constant__ sphere d_spheres_const[SCENE_1_NUM_SPHERES];
extern __constant__ material d_materials_const[SCENE_1_NUM_SPHERES];
extern __constant__ world d_world_const;


// sets hit record normal vector (see chapter 6.4)
__device__ void set_face_normal(struct hit_record& rec, const ray& r,
    const vec3& outward_normal) {
    // outward_normal is assumed to be an unit vector
    rec.front_face = dot(r.direction(), outward_normal) < 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
}

// function-based hit testing instead of virtual methods
// hit_sphere now uses the constant sphere array and material index
__device__ bool hit_sphere(const sphere& s, const ray& r,
    interval ray_t, hit_record& rec) {
    vec3 oc = s.center - r.origin();
    double a = r.direction().length_squared();
    double h = dot(r.direction(), oc);
    double c = oc.length_squared() - s.radius*s.radius;

    double discriminant = h*h - a*c;
    if (discriminant < 0) return false;

    double sqrtd = std::sqrt(discriminant);

    // find the nearest root that lies in the acceptable range.
    double root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (h + sqrtd) / a;
        if (!ray_t.surrounds(root)) return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - s.center) / s.radius;
    set_face_normal(rec,r,outward_normal);
    rec.mat_idx = s.mat_idx;

    return true;
}


// hit function for the entire world
__device__ bool hit_world(const ray& r, interval ray_t, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = ray_t.max;

    // check all spheres using the constant array and count from constant world struct
    for (int i = 0; i < d_world_const.num_spheres; i++) {
        // access sphere from the constant array
        if (hit_sphere(d_spheres_const[i], r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // add other object types as needed/desired

    return hit_anything;
}


#endif
