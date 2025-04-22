#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "vec3.h"

using color = vec3;


__device__ inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0) return sqrt(linear_component);
    return 0;
}


#endif
