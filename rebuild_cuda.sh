#!/usr/bin/bash

FLOAT_DIR=./src/FloatCUDAInOneWeekend
DOUBLE_DIR=./src/DoubleCUDAInOneWeekend

if nvcc $FLOAT_DIR/main.cu \
    -o $FLOAT_DIR/float-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_70,code=sm_70 ;
then
    echo "float-cuda-raytrace successfully built!"
fi

if nvcc $DOUBLE_DIR/main.cu \
    -o $DOUBLE_DIR/double-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_70,code=sm_70 ;
then
    echo "double-cuda-raytrace successfully built!"
fi
