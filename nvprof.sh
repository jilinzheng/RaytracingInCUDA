nvprof  --metrics inst_fp_32,inst_fp_64 \
       --trace gpu\
       ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace --scene_id 1

nvprof  --metrics inst_fp_32,inst_fp_64 \
       --trace gpu\
       ./src/ConstFloatCUDAInOneWeekend/const-float-cuda-raytrace --scene_id 1

nvprof  --metrics inst_fp_32,inst_fp_64 \
       --trace gpu\
       ./src/DoubleCUDAInOneWeekend/double-cuda-raytrace --scene_id 1

nvprof  --metrics inst_fp_32,inst_fp_64 \
       --trace gpu\
       ./src/FloatCUDAInOneWeekend/float-cuda-raytrace --scene_id 1