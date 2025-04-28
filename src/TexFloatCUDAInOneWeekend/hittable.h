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
    float t;
    bool front_face;
    int mat_idx;
};

// sphere definition (no inheritance)
struct sphere {
    point3 center;
    float radius;
    int mat_idx;

    __host__ __device__ sphere() {}
    __host__ __device__ sphere(point3 cen, float r, int mat_idx)
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


extern __constant__ world d_world_const;


// sets hit record normal vector (see chapter 6.4)
__device__ void set_face_normal(struct hit_record& rec, const ray& r,
    const vec3& outward_normal) {
    // outward_normal is assumed to be an unit vector
    rec.front_face = dot(r.direction(), outward_normal) < 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
}

// function-based hit testing instead of virtual methods
__device__ bool hit_sphere(const sphere& s, const ray& r,
    interval ray_t, hit_record& rec) {
    vec3 oc = s.center - r.origin();
    float a = r.direction().length_squared();
    float h = dot(r.direction(), oc);
    float c = oc.length_squared() - s.radius*s.radius;

    float discriminant = h*h - a*c;
    if (discriminant < 0) return false;

    float sqrtd = std::sqrt(discriminant);

    // find the nearest root that lies in the acceptable range.
    float root = (h - sqrtd) / a;
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
__device__ bool hit_world(const ray& r, interval ray_t, hit_record& rec,
    cudaTextureObject_t texObj_spheres_center_radius, cudaTextureObject_t texObj_spheres_mat_idx) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    // check all spheres
    for (int i = 0; i < d_world_const.num_spheres; i++) {
        float4 center_radius = tex1Dfetch<float4>(texObj_spheres_center_radius, i);
        int mat_idx = tex1Dfetch<int>(texObj_spheres_mat_idx, i);
        point3 center(center_radius.x, center_radius.y, center_radius.z);
        float radius = center_radius.w;

        sphere s(center,radius,mat_idx);
        if (hit_sphere(s, r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // add other object types as needed/desired

    return hit_anything;
}


#endif
