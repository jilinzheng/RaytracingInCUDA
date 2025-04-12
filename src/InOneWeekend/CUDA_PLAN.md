# Plan to Parallelize with CUDA

- kernel launched from host and executed on device need `__global__` qualifier
- function called from device and executed on device need `__device__` qualifier
- functions accessed from both host and device need both `__device__` and `__host__` qualifiers
- need to identify all the parts of `render()` and its function calls that require variables to be passed from host to device
- need to setup buffers to save all the colors of, say, a row of pixels, before transferring back to the host and actually writing out to standard output stream
  - without losing parallelization of course; calculate in device kernel, write out using host