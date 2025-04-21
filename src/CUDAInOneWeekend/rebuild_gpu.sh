#!/usr/bin/bash

nvcc ./main.cu \
    -o ./cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86
    # -g -G \

time ./cuda-raytrace > ./cuda_image.ppm

# nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./cuda-raytrace > image.ppm
