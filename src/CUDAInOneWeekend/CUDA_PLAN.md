# Plan to Parallelize with CUDA

- kernel launched from host and executed on device need `__global__` qualifier
- function called from device and executed on device need `__device__` qualifier
- functions accessed from both host and device need both `__device__` and `__host__` qualifiers
- need to identify all the parts of `render()` and its function calls that require variables to be passed from host to device; `cam.render(world)` is 'entrypoint'
  - first thing that gets called within is `initialize()`
    - sets private variables: `image_height`,`pixel_samples_scale`,`center`,`u`,`v`,`w`,`pixel_delta_u`,`pixel_delta_v`,`pixel00_loc`,`defocus_disk_u`,`defocus_disk_v`
  - in the for loops:
    - `image_height` and `image_width` is used to traverse the image dimensions, i.e. compute each pixel in the image
    - `samples_per_pixel` is public variable used by innermost loop for exactly what the name says
    - `get_ray()` uses `pixel00_loc`,`pixel_delta_u`,`pixel_delta_v`,`defocus_angle`,`center`
    - `ray_color()` calls `scatter()`
- need to setup buffers to save all the colors of, say, a row of pixels, before transferring back to the host and actually writing out to standard output stream
  - without losing parallelization of course: calculate in device kernel, write out using host
- this is a very good resource: https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/