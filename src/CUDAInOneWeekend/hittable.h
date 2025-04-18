#ifndef HITTABLE_H
#define HITTABLE_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "vec3.h"
#include "interval.h"
#include "ray.h"


// base struct for hit records (no methods)
struct hit_record {
    point3 p;
    vec3 normal;
    float t;
    bool front_face;
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
    float radius;

    __host__ __device__ sphere() {}
    __host__ __device__ sphere(point3 cen, float r) : center(cen), radius(r) {}
};

// function-based hit testing instead of virtual methods
__device__ bool hit_sphere(const sphere& s, const ray& r, 
                                   float ray_tmin, float ray_tmax, 
                                   hit_record& rec) {
    vec3 oc = s.center - r.origin();
    float a = r.direction().length_squared();
    float h = dot(r.direction(), oc);
    float c = oc.length_squared() - s.radius*s.radius;

    float discriminant = h*h - a*c;
    if (discriminant < 0) return false;

    float sqrtd = std::sqrt(discriminant);

    // find the nearest root that lies in the acceptable range.
    float root = (h - sqrtd) / a;
    if (root <= ray_tmin || ray_tmax <= root) {
        root = (h + sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - s.center) / s.radius;
    set_face_normal(rec,r,outward_normal);

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
    float ray_tmin, float ray_tmax, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_tmax;

    // check all spheres
    for (int i = 0; i < w.num_spheres; i++) {
        if (hit_sphere(w.spheres[i], r, ray_tmin, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // add other object types as needed/desired

    return hit_anything;
}


/* ORIGINAL hittable.h
class material;


class hit_record {
  public:
    point3 p;
    vec3 normal;
    shared_ptr<material> mat;
    float t;
    bool front_face;

    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class hittable {
  public:
    __device__ virtual ~hittable() = default;
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};
*/


#endif
