#!/usr/bin/bash

CONST_FLOAT_DIR=./src/ConstFloatCUDAInOneWeekend

if nvcc $CONST_FLOAT_DIR/main.cu \
    -o $CONST_FLOAT_DIR/const-float-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86 \
    -rdc=true ; # SEPARATE COMPILATION AND LINKING
then
    echo "const-float-cuda-raytrace successfully built!"
fi