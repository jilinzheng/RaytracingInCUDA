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
    camera cam;

    // try to set dimensions divisible by warp size (32)
    // and threads per row (8)

    // match the CUDA reference
    // cam.img_width    = 1280;
    // cam.img_height   = 600;

    // match serial cpu baseline/reference
    // cam.img_width    = 640;
    // cam.img_height   = 360;

    // chapter 12
    // cam.img_width   = 712;
    // cam.img_height  = 400;

    // chapter 14 final world
    // NOTE: does not fit nicely into warp size nor block dim
    // cam.img_width   = 1200;
    // cam.img_height  = 675;

    // 720p
    cam.img_width   = 1280;
    cam.img_height  = 720;

    // samples to take around a pixel for antialiasing
    cam.samples_per_pixel = 100;

    // maximum recursion depth (implemented with for-loop)
    cam.max_depth = 50;

    // positonable camera
    cam.vfov        = 20;
    cam.lookfrom    = point3(13,2,3);
    cam.lookat      = point3(0,0,0);
    cam.vup         = vec3(0,1,0);

    // defocus blur
    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    // initialize the camera given the above parameters
    cam.initialize();
    /* end image/camera configuration*/


    // total pixels
    int num_pixels = cam.img_width * cam.img_height;
    // buffer to store device-calculated pixels, to later be printed on host;
    // using Unified Memory, i.e., managed system accessible by both host and device
    // underlying implementation goes onto device global memory
    vec3 *pixel_buffer;
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&pixel_buffer, num_pixels*sizeof(vec3)));

    // square blocks to start
    int num_threads_per_block_row = 8;
    dim3 dimGrid(cam.img_width/num_threads_per_block_row,
        cam.img_height/num_threads_per_block_row);
    dim3 dimBlock(num_threads_per_block_row,num_threads_per_block_row);


    /* world creation */
    // host allocations and initializations
    // NOTE: materials are on the stack, spheres and world is on the heap
    // much less materials than spheres and world at the moment
    // in theory spheres and world could be much greater and not fit on the stack
    /* world up to chapter 11
    int num_materials = 5;
    material h_material_ground  = material(MaterialType::LAMBERTIAN, color(0.8f,0.8f,0.0f));
    material h_material_center  = material(MaterialType::LAMBERTIAN, color(0.1f,0.2f,0.5f));
    material h_material_left    = material(MaterialType::DIELETRIC, 1.50f);
    material h_material_bubble  = material(MaterialType::DIELETRIC, 1.00f/1.50f);
    material h_material_right   = material(MaterialType::METAL, color(0.8f,0.6f,0.2f), 1.0f);
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
    */

    /* chapter 12.1 world
    int num_materials = 2;
    material h_material_left    = material(MaterialType::LAMBERTIAN, color(0,0,1));
    material h_material_right   = material(MaterialType::LAMBERTIAN, color(1,0,0));
    // copy into array for convenient transfer to device
    material h_materials[] = {
        h_material_left,
        h_material_right
    };

    int num_spheres = 2;
    sphere *h_spheres = new sphere[num_spheres];
    float R = std::cos(pi/4);
    h_spheres[0] = sphere(point3(-R,0,-1), R, &h_material_left);
    h_spheres[1] = sphere(point3( R,0,-1), R, &h_material_right);

    world *h_world = new world(h_spheres,num_spheres);
    */

    // /* chapter 14 final world
    // 1 ground, 22*22 small spheres, 3 big spheres 
    int num_materials = 1+22*22+3;
    int num_spheres = num_materials;

    material *h_materials = new material[num_materials];
    sphere *h_spheres = new sphere[num_spheres];

    // ground sphere
    h_materials[0] = material(MaterialType::LAMBERTIAN, color(0.5f,0.5f,0.5f));
    h_spheres[0] = sphere(point3(0,-1000,0), 1000, &h_materials[0]);

    // small spheres
    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            float choose_mat = random_float();
            point3 center(a+0.9f*random_float(), 0.2f, b+0.9f*random_float());

            if ((center - point3(4,0.2f,0)).length() > 0.9f) {
                // scale i to start from 1 and index sequentially
                // zero-based a * total b values + zero-based b + 1
                // 1 is for the already-created ground sphere
                int i = (a+11) * 22 + (b+11) + 1;

                // diffuse
                if (choose_mat < 0.8f) {
                    color albedo    = color::random() * color::random();
                    h_materials[i]  = material(MaterialType::LAMBERTIAN, albedo);
                    h_spheres[i]    = sphere(center, 0.2f, &h_materials[i]);
                }
                // metal
                else if (choose_mat < 0.95f) {
                    color albedo    = color::random(0.5f,1.0f);
                    float fuzz      = random_float(0.0f,0.5f);
                    h_materials[i]  = material(MaterialType::METAL, albedo, fuzz);
                    h_spheres[i]    = sphere(center, 0.2f, &h_materials[i]);
                }
                // glass
                else {
                    h_materials[i]  = material(MaterialType::DIELETRIC, 1.5f);
                    h_spheres[i]    = sphere(center, 0.2f, &h_materials[i]);
                }
            }
        }
    }

    // big spheres, start index after ground and small spheres
    int i = 1+22*22;
    h_materials[i] = material(MaterialType::DIELETRIC, 1.5f);
    h_spheres[i]   = sphere(point3(0,1,0), 1.0f, &h_materials[i]);

    h_materials[i+1] = material(MaterialType::LAMBERTIAN, color(0.4f,0.2f,0.1f));
    h_spheres[i+1]   = sphere(point3(-4,1,0), 1.0f, &h_materials[i+1]);

    h_materials[i+2] = material(MaterialType::METAL, color(0.7f,0.6f,0.5f), 0.0f);
    h_spheres[i+2]   = sphere(point3(4,1,0), 1.0f, &h_materials[i+2]);

    world *h_world = new world(h_spheres,num_spheres);
    // */

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
    init_rng<<<dimGrid, dimBlock>>>(cam.img_width, cam.img_height, d_rand_state);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // call the render() kernel
    render<<<dimGrid, dimBlock>>>(pixel_buffer, cam, d_world, d_rand_state);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // output pixel_buffer as a .ppm image
    const interval intensity(0.000f,0.999f);
    std::cout << "P3\n" << cam.img_width << " " << cam.img_height << "\n255\n";
    for (int j = 0; j < cam.img_height; ++j) {      // rows
        for (int i = 0; i < cam.img_width; ++i) {   // cols
            size_t pixel_index = j * cam.img_width + i;
            vec3 pixel = pixel_buffer[pixel_index];
            int r = int(256 * intensity.clamp(pixel.x()));
            int g = int(256 * intensity.clamp(pixel.y()));
            int b = int(256 * intensity.clamp(pixel.z()));
            std::cout << r << " " << g << " " << b << "\n";
        }
    }

    // cudaFree device allocations, delete host heap allocations
    CUDA_SAFE_CALL(cudaFree(pixel_buffer));
    CUDA_SAFE_CALL(cudaFree(d_materials));
    CUDA_SAFE_CALL(cudaFree(d_spheres));
    CUDA_SAFE_CALL(cudaFree(d_world));
    CUDA_SAFE_CALL(cudaFree(d_rand_state));
    delete[] h_materials;
    delete[] h_spheres;
    delete h_world;

    return 0;
}
