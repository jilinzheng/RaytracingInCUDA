#!/usr/bin/bash

FLOAT_DIR=./src/GlobalFloatCUDAInOneWeekend
DOUBLE_DIR=./src/GlobalDoubleCUDAInOneWeekend

if nvcc $FLOAT_DIR/main.cu \
    -o $FLOAT_DIR/global-float-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_70,code=sm_70 ;
then
    echo "global-float-cuda-raytrace successfully built!"
fi

if nvcc $DOUBLE_DIR/main.cu \
    -o $DOUBLE_DIR/global-double-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_70,code=sm_70 ;
then
    echo "global-double-cuda-raytrace successfully built!"
fi
