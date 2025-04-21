#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <curand_kernel.h>


// constants
const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;


// utility Functions
__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__device__ inline float random_float(curandState *thread_rand_state) {
    // returns a random real in [0,1).
    // return std::rand() / (RAND_MAX + 1.0f);
    return curand_uniform(thread_rand_state);
}

__device__ inline float random_float(float min, float max, curandState *thread_rand_state) {
    // returns a random real in [min,max).
    return min + (max-min)*random_float(thread_rand_state);
}

// initialize random states for device (call this before generating random numbers)
__global__ void init_rng(int img_width, int img_height, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= img_width) || (j >= img_height)) return;
    int pixel_index = j * img_width + i;
    // each thread gets same seed, a different sequence number, no offset
    curand_init(1227, pixel_index, 0, &rand_state[pixel_index]);
}


#endif
