# nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace --scene_id 1

# nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./src/ConstFloatCUDAInOneWeekend/const-float-cuda-raytrace --scene_id 1

# nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./src/DoubleCUDAInOneWeekend/double-cuda-raytrace --scene_id 1

# nvprof  --metrics inst_fp_32,inst_fp_64 \
#        --trace gpu\
#        ./src/FloatCUDAInOneWeekend/float-cuda-raytrace --scene_id 1


# ncu intended to be used on Volta and newer (compute capability 7.0 and higher) GPUs only

ncu --set detailed -k render -o const-double-render-profile ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace --scene_id 1

ncu --set detailed -k render -o const-double-render-profile ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace --scene_id 1

ncu --set detailed -k render -o const-double-render-profile ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace --scene_id 1

ncu --set detailed -k render -o const-double-render-profile ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace --scene_id 1


ncu -k render --metrics smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum \
    ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace --scene_id 1

ncu -k render --metrics smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum \
    ./src/DoubleCUDAInOneWeekend/double-cuda-raytrace --scene_id 1

ncu -k render --metrics smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum \
    ./src/ConstFloatCUDAInOneWeekend/const-float-cuda-raytrace --scene_id 1

ncu -k render --metrics smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum \
    ./src/FloatCUDAInOneWeekend/float-cuda-raytrace --scene_id 1

# NOTE: curandState = curandStateXORWOW has double in the struct,
# so profiling the float implementations still have some fp64 used...
# not sure why the double implementations have so many fp32 though,
# but they certainly have a lot of fp64 used
