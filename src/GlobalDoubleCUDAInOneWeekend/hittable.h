#ifndef HITTABLE_H
#define HITTABLE_H

#include "vec3.h"
#include "ray.h"
#include "interval.h"


struct material;

// base struct for hit records (no methods)
struct hit_record {
    point3 p;
    vec3 normal;
    double t;
    bool front_face;
    material *mat;
};

// sets hit record normal vector (see chapter 6.4)
__device__ void set_face_normal(struct hit_record& rec, const ray& r,
    const vec3& outward_normal) {
    // outward_normal is assumed to be an unit vector
    rec.front_face = dot(r.direction(), outward_normal) < 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
}

// sphere definition (no inheritance)
struct sphere {
    point3 center;
    double radius;
    material *mat;

    __host__ __device__ sphere() {}
    __host__ __device__ sphere(point3 cen, double r, material *mat)
        : center(cen), radius(r), mat(mat) {}
};

// function-based hit testing instead of virtual methods
__device__ bool hit_sphere(const sphere& s, const ray& r,
    interval ray_t, hit_record& rec) {
    vec3 oc = s.center - r.origin();
    double a = r.direction().length_squared();
    double h = dot(r.direction(), oc);
    double c = oc.length_squared() - s.radius*s.radius;

    double discriminant = h*h - a*c;
    if (discriminant < 0) return false;

    double sqrtd = sqrt(discriminant);

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
    rec.mat = s.mat;

    return true;
}

// world consists of sphere and any other objects in the future
// analogous to hittable_list - chapter 6.5
struct world {
    sphere* spheres;
    int num_spheres;
    // add other object types as needed

    __host__ __device__ world() : spheres(nullptr), num_spheres(0) {}
    __host__ __device__ world(sphere* s, int ns) : spheres(s), num_spheres(ns) {}
};

// hit function for the entire world
__device__ bool hit_world(const world& w, const ray& r,
    interval ray_t, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = ray_t.max;

    // check all spheres
    for (int i = 0; i < w.num_spheres; i++) {
        if (hit_sphere(w.spheres[i], r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // add other object types as needed/desired

    return hit_anything;
}


#endif
