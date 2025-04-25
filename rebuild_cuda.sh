#!/usr/bin/bash

FLOAT_DIR=./src/FloatCUDAInOneWeekend
DOUBLE_DIR=./src/DoubleCUDAInOneWeekend

nvcc $FLOAT_DIR/main_global_mem.cu \
    -o $FLOAT_DIR/float-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86

echo "float-cuda-raytrace successfully built!"

nvcc $DOUBLE_DIR/main.cu \
    -o $DOUBLE_DIR/double-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86

echo "double-cuda-raytrace successfully built!"

# nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./cuda-raytrace > image.ppm
