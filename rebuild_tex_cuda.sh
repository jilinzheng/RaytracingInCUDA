#!/usr/bin/bash

TEX_FLOAT_DIR=./src/TexFloatCUDAInOneWeekend
# CONST_DOUBLE_DIR=./src/ConstDoubleCUDAInOneWeekend

if nvcc $TEX_FLOAT_DIR/main.cu \
    -o $TEX_FLOAT_DIR/tex-float-cuda-raytrace \
    -O3 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_70,code=sm_70 \
    --ptxas-options=-v \
    -rdc=true ; # SEPARATE COMPILATION AND LINKING
then
    echo "tex-float-cuda-raytrace successfully built!"
fi

# if nvcc $CONST_DOUBLE_DIR/main.cu \
#     -o $CONST_DOUBLE_DIR/const-double-cuda-raytrace \
#     -O3 \
#     -gencode arch=compute_86,code=sm_86 \
#     -gencode arch=compute_70,code=sm_70 \
#     -rdc=true ; # SEPARATE COMPILATION AND LINKING
# then
#     echo "const-double-cuda-raytrace successfully built!"
# fi
