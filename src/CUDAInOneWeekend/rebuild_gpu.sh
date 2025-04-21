#!/usr/bin/bash

nvcc ./main.cu \
    -o ./cuda-raytrace \
    -g -G \
    -Wno-deprecated-gpu-targets

# ./cuda-raytrace > ./image.ppm

# nsys profile --output cuda_raytrace_timeline ./cuda-raytrace > image.ppm

ncu --metrics inst_fp_32,inst_fp_64 --output cuda_raytrace_metrics ./cuda-raytrace > image.ppm

eog ./image.ppm

#nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./cuda-raytrace > image.ppm
