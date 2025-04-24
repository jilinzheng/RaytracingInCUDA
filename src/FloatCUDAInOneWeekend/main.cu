#include "rtweekend.h"
#include "hittable.h"
#include "color.h"
#include "camera.h"
#include "curand_kernel.h"
#include "material.h"
#include <iostream>
#include <iomanip>
#include "cxxopts.hpp"
#include <fstream>
#include <sstream>

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

int main(int argc, char* argv[]) {
    /* begin parsing */
    cxxopts::Options options("./cuda-raytrace",
        "Super Raytrace: Raytracing with CUDA");

    options.add_options()
        ("scene_id", "ID of the scene to render",
            cxxopts::value<int>())
        ("width", "Width of the output image",
            cxxopts::value<int>()->default_value("320"))
        ("height", "Height of the output image",
            cxxopts::value<int>()->default_value("192"))
        ("samples", "Number of samples per pixel",
            cxxopts::value<int>()->default_value("10"))
        ("bounces", "Maximum number of ray bounces",
            cxxopts::value<int>()->default_value("25"))
        ("threads_per_2d_block_row", "Number of threads per 2-D thread block.",
            cxxopts::value<int>()->default_value("8"))
        ("h,help", "Print usage"); // Added a help flag

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    // check if scene_id was provided as it doesn't have a default value
    if (!result.count("scene_id")) {
        std::cerr << "Error: --scene_id is required." << "\n";
        std::cout << options.help() << "\n";
        return 1;
    }

    int scene_id = result["scene_id"].as<int>();
    int width = result["width"].as<int>();
    int height = result["height"].as<int>();
    int samples = result["samples"].as<int>();
    int bounces = result["bounces"].as<int>();
    int threads_per_2d_block_row = result["threads_per_2d_block_row"].as<int>();
    /* end parsing */


    // select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    /* timing setup */
    cudaEvent_t render_only_start, render_only_stop;
    cudaEvent_t end_to_end_start, end_to_end_stop;
    float render_only_elapsed, end_to_end_elapsed;

    // create cuda events
    cudaEventCreate(&render_only_start);
    cudaEventCreate(&render_only_stop);
    cudaEventCreate(&end_to_end_start);
    cudaEventCreate(&end_to_end_stop);

    // start end-to-end timing
    cudaEventRecord(end_to_end_start, 0);
    /* end timing setup */


    /* image/camera configuration */
    camera cam;

    // try to set dimensions divisible by warp size (32)
    // and threads per row (8)
    cam.img_width   = width;
    cam.img_height  = height;

    // samples to take around a pixel for antialiasing
    cam.samples_per_pixel = samples;

    // maximum recursion depth (implemented with for-loop)
    cam.max_depth = bounces;

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
    dim3 dimGrid(cam.img_width/threads_per_2d_block_row,
        cam.img_height/threads_per_2d_block_row);
    dim3 dimBlock(threads_per_2d_block_row,threads_per_2d_block_row);


    /* world creation as determined by scene_id */
    // host allocations and initializations
    int num_materials, num_spheres;
    material *h_materials;
    sphere *h_spheres;

    switch (scene_id) {
        // original scene from end of book
        case 1: {
            // 1 ground, 22*22 small spheres, 3 big spheres
            num_materials = 1+22*22+3;
            num_spheres = num_materials;

            h_materials = new material[num_materials];
            h_spheres = new sphere[num_spheres];

            // ground sphere
            h_materials[0] = material(MaterialType::LAMBERTIAN, color(0.5,0.5,0.5));
            h_spheres[0] = sphere(point3(0,-1000,0), 1000, &h_materials[0]);

            // small spheres
            for (int a = -11; a < 11; ++a) {
                for (int b = -11; b < 11; ++b) {
                    float choose_mat = random_float();
                    point3 center(a+0.9*random_float(), 0.2, b+0.9*random_float());

                    if ((center - point3(4,0.2,0)).length() > 0.9) {
                        // scale i to start from 1 and index sequentially
                        // zero-based a * total b values + zero-based b + 1
                        // 1 is for the already-created ground sphere
                        int i = (a + 11) * 22 + (b + 11) + 1;

                        // diffuse
                        if (choose_mat < 0.8) {
                            color albedo    = color::random() * color::random();
                            h_materials[i]  = material(MaterialType::LAMBERTIAN, albedo);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                        // metal
                        else if (choose_mat < 0.95) {
                            color albedo    = color::random(0.5,1.0);
                            float fuzz      = random_float(0.0,0.5);
                            h_materials[i]  = material(MaterialType::METAL, albedo, fuzz);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                        // glass
                        else {
                            h_materials[i]  = material(MaterialType::DIELETRIC, 1.5);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                    }
                }
            }
            break;
        }
        case 2: {
            // 1 ground, 6*6 small spheres, 3 big spheres
            num_materials = 1+6*6+3;
            num_spheres = num_materials;

            h_materials = new material[num_materials];
            h_spheres = new sphere[num_spheres];

            // ground sphere
            h_materials[0] = material(MaterialType::LAMBERTIAN, color(0.5,0.5,0.5));
            h_spheres[0] = sphere(point3(0,-1000,0), 1000, &h_materials[0]);

            // small spheres
            for (int a = 5; a < 11; a++) {
                for (int b = 5; b < 11; b++) {
                    float choose_mat = random_float();
                    point3 center(a+0.9*random_float(), 0.2, b+0.9*random_float());

                    if ((center - point3(4,0.2,0)).length() > 0.9) {
                        int i = (a - 5) * 6 + (b - 5) + 1;

                        // diffuse
                        if (choose_mat < 0.8) {
                            color albedo    = color::random() * color::random();
                            h_materials[i]  = material(MaterialType::LAMBERTIAN, albedo);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                        // metal
                        else if (choose_mat < 0.95) {
                            color albedo    = color::random(0.5,1.0);
                            float fuzz      = random_float(0.0,0.5);
                            h_materials[i]  = material(MaterialType::METAL, albedo, fuzz);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                        // glass
                        else {
                            h_materials[i]  = material(MaterialType::DIELETRIC, 1.5);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                    }
                }
            }
            break;
        }
        default: {
            // 1 ground, 11*11 small spheres, 3 big spheres
            num_materials = 1+11*11+3;
            num_spheres = num_materials;

            h_materials = new material[num_materials];
            h_spheres = new sphere[num_spheres];

            // ground sphere
            h_materials[0] = material(MaterialType::LAMBERTIAN, color(0.5,0.5,0.5));
            h_spheres[0] = sphere(point3(0,-1000,0), 1000, &h_materials[0]);

            for (int a = -11; a < 0; a++) {
                for (int b = -11; b < 0; b++) {
                    float choose_mat = random_float();
                    point3 center(a+0.9*random_float(), 0.2, b+0.9*random_float());

                    if ((center - point3(4,0.2,0)).length() > 0.9) {
                        int i = (a + 11) * 11 + (b + 11) + 1;

                        // diffuse
                        if (choose_mat < 0.8) {
                            color albedo    = color::random() * color::random();
                            h_materials[i]  = material(MaterialType::LAMBERTIAN, albedo);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                        // metal
                        else if (choose_mat < 0.95) {
                            color albedo    = color::random(0.5,1.0);
                            float fuzz      = random_float(0.0,0.5);
                            h_materials[i]  = material(MaterialType::METAL, albedo, fuzz);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                        // glass
                        else {
                            h_materials[i]  = material(MaterialType::DIELETRIC, 1.5);
                            h_spheres[i]    = sphere(center, 0.2, &h_materials[i]);
                        }
                    }
                }
            }
            break;
        }
    }

    // shared 3 big spheres, start index after ground and small spheres
    int i = num_spheres-3;
    // middle sphere
    h_materials[i] = material(MaterialType::DIELETRIC, 1.5);
    h_spheres[i]   = sphere(point3(0,1,0), 1.0, &h_materials[i]);
    // rear sphere
    h_materials[i+1] = material(MaterialType::LAMBERTIAN, color(0.4,0.2,0.1));
    h_spheres[i+1]   = sphere(point3(-4,1,0), 1.0, &h_materials[i+1]);
    // front sphere
    h_materials[i+2] = material(MaterialType::METAL, color(0.7,0.6,0.5), 0.0);
    h_spheres[i+2]   = sphere(point3(4,1,0), 1.0, &h_materials[i+2]);

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
    curandState *d_rand_states;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rand_states, num_pixels*sizeof(curandState)));
    init_rng<<<dimGrid, dimBlock>>>(cam.img_width, cam.img_height, d_rand_states);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // call the render() kernel
    // start (render) kernel-only timing
    cudaEventRecord(render_only_start, 0);
    render<<<dimGrid, dimBlock>>>(pixel_buffer, cam, d_world, d_rand_states);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // stop (render) kernel-only timing
    cudaEventRecord(render_only_stop,0);
    cudaEventSynchronize(render_only_stop);
    cudaEventElapsedTime(&render_only_elapsed, render_only_start, render_only_stop);
    std::cout << std::fixed << std::setprecision(8)
    << std::setw(15) << render_only_elapsed<< ",";
    cudaEventDestroy(render_only_start);
    cudaEventDestroy(render_only_stop);

    /* begin writing pixel_buffer out to .ppm file */
    // construct filename
    std::stringstream f_ss;
    f_ss
        << "./benchmarks/float_global_mem_ppms/"
        << "scene" << scene_id
        << "_" << width << "x" << height
        << "_" << samples << "samples"
        << "_" << bounces << "bounces"
        << ".ppm";
    std::string f = f_ss.str();

    // open file for writing
    std::ofstream ofs(f);
    if (!ofs) {
        std::cerr << "Error: Could not open file for writing: " << f << "\n";
        return -1;
    }

    // output pixel_buffer as a .ppm image
    const interval intensity(0.000,0.999);
    ofs << "P3\n" << cam.img_width << " " << cam.img_height << "\n255\n";
    for (int j = 0; j < cam.img_height; ++j) {      // rows
        for (int i = 0; i < cam.img_width; ++i) {   // cols
            size_t pixel_index = j * cam.img_width + i;
            vec3 pixel = pixel_buffer[pixel_index];
            int r = int(256 * intensity.clamp(pixel.x()));
            int g = int(256 * intensity.clamp(pixel.y()));
            int b = int(256 * intensity.clamp(pixel.z()));
            ofs << r << " " << g << " " << b << "\n";
        }
    }
    /* end writing */


    // cudaFree device allocations, delete host heap allocations
    CUDA_SAFE_CALL(cudaFree(pixel_buffer));
    CUDA_SAFE_CALL(cudaFree(d_materials));
    CUDA_SAFE_CALL(cudaFree(d_spheres));
    CUDA_SAFE_CALL(cudaFree(d_world));
    CUDA_SAFE_CALL(cudaFree(d_rand_states));
    delete[] h_materials;
    delete[] h_spheres;
    delete h_world;

    // stop end-to-end timing
    cudaEventRecord(end_to_end_stop,0);
    cudaEventSynchronize(end_to_end_stop);
    cudaEventElapsedTime(&end_to_end_elapsed, end_to_end_start, end_to_end_stop);
    std::cout << std::fixed << std::setprecision(8)
    << std::setw(15) << end_to_end_elapsed << "\n";
    cudaEventDestroy(end_to_end_start);
    cudaEventDestroy(end_to_end_stop);

    return 0;
}
