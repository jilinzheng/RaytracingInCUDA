// Note: Watch out! Include order will matter!
#include "rtweekend.h"
#include "hittable.h"
#include "color.h"
#include "camera.h"
#include "curand_kernel.h"
#include "material.h"

// assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void update_world_pointer(world *w, sphere *spheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        w->spheres = spheres;
    }
}

__global__ void update_material_pointers(sphere* d_spheres, material* d_materials, int num_spheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < num_spheres; ++i)
            d_spheres[i].mat = &d_materials[i];
    }
}

int main() {
    // select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    /* image/camera configuration */
    // these dimensions match the CUDA reference
    // int img_width = 1280, img_height = 600;
    // these dimensions match serial cpu baseline/reference
    int img_width = 640, img_height = 360;
    // both are divisible by warp size (32) and threads per row (8)

    // total pixels
    int num_pixels = img_width*img_height;

    // samples to take around a pixel for antialiasing
    int samples_per_pixel = 100;

    // maximum recursion depth (implemented with for-loop)
    int max_depth = 50;

    // initialize the camera
    camera cam(img_width,img_height,samples_per_pixel,max_depth);
    /* end image/camera configuration*/


    // buffer to store device-calculated pixels, to later be printed on host;
    // using Unified Memory, i.e., managed system accessible by both host and device
    // underlying implementation goes onto device global memory
    vec3 *pixel_buffer;
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&pixel_buffer, num_pixels*sizeof(vec3)));

    // square blocks to start
    int num_threads_per_block_row = 8;
    dim3 dimGrid(img_width/num_threads_per_block_row,
        img_height/num_threads_per_block_row);
    dim3 dimBlock(num_threads_per_block_row,num_threads_per_block_row);


    /* world creation */
    // host allocations and initializations
    // NOTE: materials are on the stack, spheres and world is on the heap
    // much less materials than spheres and world at the moment
    // in theory spheres and world could be much greater and not fit on the stack
    int num_materials = 5;
    material h_material_ground  = material(MaterialType::LAMBERTIAN, color(0.8f,0.8f,0.0f));
    material h_material_center  = material(MaterialType::LAMBERTIAN, color(0.1f,0.2f,0.5f));
    material h_material_left    = material(MaterialType::DIELETRIC, 1.50f);
    material h_material_bubble  = material(MaterialType::DIELETRIC, 1.00f/1.50f);
    material h_material_right   = material(MaterialType::METAL, color(0.8f,0.6f,0.2f), 0.0f);
    // copy into array for convenient transfer to device
    material h_materials[] = {
        h_material_ground,
        h_material_center,
        h_material_left,
        h_material_bubble,
        h_material_right
    };
    // int num_materials = sizeof(h_materials) / sizeof(h_materials[0]);

    int num_spheres = 5;
    sphere *h_spheres = new sphere[num_spheres];
    h_spheres[0] = sphere(point3( 0.0f, -100.5f, -1.0f), 100.0f, &h_material_ground);
    h_spheres[1] = sphere(point3( 0.0f,    0.0f, -1.2f),   0.5f, &h_material_center);
    h_spheres[2] = sphere(point3(-1.0f,    0.0f, -1.0f),   0.5f, &h_material_left);
    h_spheres[3] = sphere(point3(-1.0f,    0.0f, -1.0f),   0.4f, &h_material_bubble);
    h_spheres[4] = sphere(point3( 1.0f,    0.0f, -1.0f),   0.5f, &h_material_right);

    world *h_world = new world(h_spheres,num_spheres);

    // device allocations and transfers
    material *d_materials;
    CUDA_SAFE_CALL(cudaMalloc(&d_materials,num_materials*sizeof(material)));
    CUDA_SAFE_CALL(cudaMemcpy(d_materials,h_materials,num_materials*sizeof(material),
        cudaMemcpyHostToDevice));

    sphere *d_spheres;
    CUDA_SAFE_CALL(cudaMalloc(&d_spheres, num_spheres*sizeof(sphere)));
    CUDA_SAFE_CALL(cudaMemcpy(d_spheres,h_spheres,num_spheres*sizeof(sphere),
        cudaMemcpyHostToDevice));

    world *d_world;
    CUDA_SAFE_CALL(cudaMalloc(&d_world,sizeof(world)));
    CUDA_SAFE_CALL(cudaMemcpy(d_world,h_world,sizeof(world),
        cudaMemcpyHostToDevice));

    // update world and material pointers since host pointers are invalid after transfer
    // after transferring to device
    update_world_pointer<<<1,1>>>(d_world, d_spheres);
    update_material_pointers<<<1,1>>>(d_spheres, d_materials, num_spheres);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    /* end world creation*/


    // setup random number generation in device
    curandState *d_rand_state;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    init_rng<<<dimGrid, dimBlock>>>(img_width, img_height, d_rand_state);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // call the render() kernel
    render<<<dimGrid, dimBlock>>>(pixel_buffer, cam, d_world, d_rand_state);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // output pixel_buffer as a .ppm image
    const interval intensity(0.000f,0.999f);
    std::cout << "P3\n" << img_width << " " << img_height << "\n255\n";
    for (int j = 0; j < img_height; ++j) {      // rows
        for (int i = 0; i < img_width; ++i) {   // cols
            size_t pixel_index = j * img_width + i;
            vec3 pixel = pixel_buffer[pixel_index];
            int r = int(256 * intensity.clamp(pixel.x()));
            int g = int(256 * intensity.clamp(pixel.y()));
            int b = int(256 * intensity.clamp(pixel.z()));
            std::cout << r << " " << g << " " << b << "\n";
        }
    }

    // cudaFree and delete heap allocations
    CUDA_SAFE_CALL(cudaFree(pixel_buffer));
    CUDA_SAFE_CALL(cudaFree(d_materials));
    CUDA_SAFE_CALL(cudaFree(d_spheres));
    CUDA_SAFE_CALL(cudaFree(d_world));
    CUDA_SAFE_CALL(cudaFree(d_rand_state));
    delete h_spheres;
    delete h_world;

    return 0;
}
