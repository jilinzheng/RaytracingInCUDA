#ifndef RAY_H
#define RAY_H


#include "vec3.h"


class ray {
  public:
    __device__ ray() {}
    __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    // return immutable references to members
    __device__ const point3& origin() const  { return orig; }
    __device__ const vec3& direction() const { return dir; }

    // P(t) = A + t*b, where A is ray origin, b is ray direction,
    // and t is ray parameter
    __device__ point3 at(double t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    vec3 dir;
};


#endif
