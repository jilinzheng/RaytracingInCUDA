#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>


// constants
const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;


// utility Functions
__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__host__ __device__ inline float random_float() {
    // returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0f);
}

__host__ __device__ inline float random_float(float min, float max) {
    // returns a random real in [min,max).
    return min + (max-min)*random_float();
}


#endif
