#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "vec3.h"

using color = vec3;


__device__ inline float linear_to_gamma(float linear_component) {
    if (linear_component > 0) return sqrtf(linear_component);
    return 0;
}


#endif
