#!/usr/bin/bash

nvcc ./main.cu \
    -o ./cuda-raytrace \
    -g -G \
    -Wno-deprecated-gpu-targets

./cuda-raytrace > ./image.ppm

#nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./cuda-raytrace > image.ppm
