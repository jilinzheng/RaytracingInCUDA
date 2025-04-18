#!/usr/bin/bash

nvcc /usr4/ugrad/jilin/ec527/super-raytrace/src/CUDAInOneWeekend/main.cu \
    -o /usr4/ugrad/jilin/ec527/super-raytrace/src/CUDAInOneWeekend/cuda-raytrace \
    -g -G

# nvprof --metrics inst_fp_32,inst_fp_64 ./cuda-raytrace > image.ppm