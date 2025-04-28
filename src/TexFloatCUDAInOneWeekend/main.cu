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


__constant__ world d_world_const;


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
        ("threads", "Number of threads per 2-D thread block row.",
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
    int threads_per_2d_block_row = result["threads"].as<int>();

    // Ensure only scene 1 is selected for this texture memory version
    if (scene_id != 1) {
        std::cerr << "Error: This version only supports scene_id 1 with constant memory." << std::endl;
        return 1;
    }
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

    // 1 ground, 22*22 small spheres, 3 big spheres
    num_materials = 1+22*22+3;
    num_spheres = num_materials;

    h_materials = new material[num_materials];
    h_spheres = new sphere[num_spheres];

    // ground sphere
    h_materials[0] = material(MaterialType::LAMBERTIAN, color(0.5,0.5,0.5));
    h_spheres[0] = sphere(point3(0,-1000,0), 1000, 0);

    // small spheres
    int sphere_idx = 1;
    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            float choose_mat = random_float();
            point3 center(a+0.9*random_float(), 0.2, b+0.9*random_float());

            if ((center - point3(4,0.2,0)).length() > 0.9) {
                int h_idx = sphere_idx;

                // ensure h_idx doesn't exceed small spheres
                // NOTE: this check may not be necessary
                if (h_idx < (1+22*22)) {
                    // diffuse
                    if (choose_mat < 0.8) {
                        color albedo    = color::random() * color::random();
                        h_materials[sphere_idx]  = material(MaterialType::LAMBERTIAN, albedo);
                        h_spheres[sphere_idx]    = sphere(center, 0.2, h_idx);
                    }
                    // metal
                    else if (choose_mat < 0.95) {
                        color albedo    = color::random(0.5,1.0);
                        float fuzz      = random_float(0.0,0.5);
                        h_materials[h_idx]  = material(MaterialType::METAL, albedo, fuzz);
                        h_spheres[h_idx]    = sphere(center, 0.2, h_idx);
                    }
                    // glass
                    else {
                        h_materials[h_idx]  = material(MaterialType::DIELETRIC, 1.5);
                        h_spheres[h_idx]    = sphere(center, 0.2, h_idx);
                    }
                    ++sphere_idx; // only increment when a sphere is added
                }
            }
        }
    }

    // shared 3 big spheres, start index after ground and small spheres
    int i = num_spheres-3;
    // middle sphere
    h_materials[i] = material(MaterialType::DIELETRIC, 1.5);
    h_spheres[i]   = sphere(point3(0,1,0), 1.0, i);
    // rear sphere
    h_materials[i+1] = material(MaterialType::LAMBERTIAN, color(0.4,0.2,0.1));
    h_spheres[i+1]   = sphere(point3(-4,1,0), 1.0, i+1);
    // front sphere
    h_materials[i+2] = material(MaterialType::METAL, color(0.7,0.6,0.5), 0.0);
    h_spheres[i+2]   = sphere(point3(4,1,0), 1.0, i+2);

    world h_world(num_spheres);

    // device allocations and transfers
    // world in constant memory; only need number of spheres from it
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_world_const, &h_world, sizeof(world)));

    // material *d_materials;
    // CUDA_SAFE_CALL(cudaMalloc(&d_materials,num_materials*sizeof(material)));
    // CUDA_SAFE_CALL(cudaMemcpy(d_materials,h_materials,num_materials*sizeof(material),
    //     cudaMemcpyHostToDevice));

    // sphere *d_spheres;
    // CUDA_SAFE_CALL(cudaMalloc(&d_spheres, num_spheres*sizeof(sphere)));
    // CUDA_SAFE_CALL(cudaMemcpy(d_spheres,h_spheres,num_spheres*sizeof(sphere),
    //     cudaMemcpyHostToDevice));

    // create texture objects for the allocations
    // pack material and spheres values into simpler structures for texture fetching
    std::vector<float4> h_materials_albedo_fuzz(num_materials);
    std::vector<float> h_materials_refraction_index(num_materials);
    std::vector<int> h_materials_type(num_materials);
    std::vector<float4> h_spheres_center_radius(num_spheres);
    std::vector<int> h_spheres_mat_idx(num_spheres);

    // TODO: can combine the loops
    for (size_t i = 0; i < num_materials; ++i) {
        h_materials_albedo_fuzz[i] = make_float4(h_materials[i].albedo.x(),h_materials[i].albedo.y(),
            h_materials[i].albedo.z(), h_materials[i].fuzz);
        h_materials_refraction_index[i] = h_materials[i].refraction_index;
        h_materials_type[i] = static_cast<int>(h_materials[i].type);
    }

    for (int i = 0; i < num_spheres; ++i) {
        h_spheres_center_radius[i] = make_float4(h_spheres[i].center.x(), h_spheres[i].center.y(),
            h_spheres[i].center.z(), h_spheres[i].radius);
        h_spheres_mat_idx[i] = h_spheres[i].mat_idx;
    }

    float4 *d_materials_albedo_fuzz;
    float *d_materials_refraction_index;
    int *d_materials_type;
    float4 *d_spheres_center_radius;
    int    *d_spheres_mat_idx;

    CUDA_SAFE_CALL(cudaMalloc(&d_materials_albedo_fuzz, h_materials_albedo_fuzz.size() * sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpy(d_materials_albedo_fuzz, h_materials_albedo_fuzz.data(),
        h_materials_albedo_fuzz.size() * sizeof(float4), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&d_materials_refraction_index, h_materials_refraction_index.size() * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_materials_refraction_index, h_materials_refraction_index.data(),
        h_materials_refraction_index.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&d_materials_type, h_materials_type.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_materials_type, h_materials_type.data(),
        h_materials_type.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&d_spheres_center_radius, num_spheres * sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemcpy(d_spheres_center_radius, h_spheres_center_radius.data(),
        num_spheres * sizeof(float4), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&d_spheres_mat_idx, num_spheres * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_spheres_mat_idx, h_spheres_mat_idx.data(),
        num_spheres * sizeof(int), cudaMemcpyHostToDevice));

    // materials (1D)
    // cudaResourceDesc resDesc_materials = {};
    // resDesc_materials.resType = cudaResourceTypeLinear;
    // resDesc_materials.res.linear.devPtr = d_materials;
    // resDesc_materials.res.linear.sizeInBytes = num_materials * sizeof(material);
    // resDesc_materials.res.linear.desc = cudaCreateChannelDesc<material>();
    // cudaTextureDesc texDesc_materials = {};
    // texDesc_materials.readMode = cudaReadModeElementType;
    // cudaTextureObject_t texObj_materials;
    // CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj_materials,&resDesc_materials,&texDesc_materials,nullptr));

    // Texture for Albedo (float3) and Fuzz (float) - stored as float4
    cudaResourceDesc resDesc_albedo_fuzz = {};
    resDesc_albedo_fuzz.resType = cudaResourceTypeLinear;
    resDesc_albedo_fuzz.res.linear.devPtr = d_materials_albedo_fuzz;
    resDesc_albedo_fuzz.res.linear.sizeInBytes = h_materials_albedo_fuzz.size() * sizeof(float4);
    resDesc_albedo_fuzz.res.linear.desc = cudaCreateChannelDesc<float4>(); // Element type is float4
    cudaTextureDesc texDesc_albedo_fuzz = {};
    texDesc_albedo_fuzz.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj_materials_albedo_fuzz;
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj_materials_albedo_fuzz, &resDesc_albedo_fuzz,
        &texDesc_albedo_fuzz, nullptr));

    // Texture for Refraction Index (float) - stored as float
    cudaResourceDesc resDesc_ri = {};
    resDesc_ri.resType = cudaResourceTypeLinear;
    resDesc_ri.res.linear.devPtr = d_materials_refraction_index;
    resDesc_ri.res.linear.sizeInBytes = h_materials_refraction_index.size() * sizeof(float);
    resDesc_ri.res.linear.desc = cudaCreateChannelDesc<float>(); // Element type is float
    cudaTextureDesc texDesc_ri = {};
    texDesc_ri.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj_materials_ri;
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj_materials_ri, &resDesc_ri, &texDesc_ri, nullptr));

    // Texture for Material Type (int) - stored as int
    cudaResourceDesc resDesc_type = {};
    resDesc_type.resType = cudaResourceTypeLinear;
    resDesc_type.res.linear.devPtr = d_materials_type;
    resDesc_type.res.linear.sizeInBytes = h_materials_type.size() * sizeof(int);
    resDesc_type.res.linear.desc = cudaCreateChannelDesc<int>(); // Element type is int
    cudaTextureDesc texDesc_type = {};
    texDesc_type.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj_materials_type;
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj_materials_type, &resDesc_type, &texDesc_type, nullptr));

    // spheres (1D)
    // cudaResourceDesc resDesc_spheres = {};
    // resDesc_spheres.resType = cudaResourceTypeLinear;
    // resDesc_spheres.res.linear.devPtr = d_spheres;
    // resDesc_spheres.res.linear.sizeInBytes = num_spheres * sizeof(sphere);
    // resDesc_spheres.res.linear.desc = cudaCreateChannelDesc<sphere>();
    // cudaTextureDesc texDesc_spheres = {};
    // texDesc_spheres.readMode = cudaReadModeElementType;
    // cudaTextureObject_t texObj_spheres;
    // CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj_spheres, &resDesc_spheres, &texDesc_spheres, nullptr));

    // Texture for Sphere Position (float3) and Radius (float) - stored as float4
    cudaResourceDesc resDesc_center_radius = {};
    resDesc_center_radius.resType = cudaResourceTypeLinear;
    resDesc_center_radius.res.linear.devPtr = d_spheres_center_radius;
    resDesc_center_radius.res.linear.sizeInBytes = num_spheres * sizeof(float4);
    resDesc_center_radius.res.linear.desc = cudaCreateChannelDesc<float4>(); // Element type is float4
    cudaTextureDesc texDesc_pos_radius = {};
    texDesc_pos_radius.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj_spheres_center_radius;
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj_spheres_center_radius, &resDesc_center_radius,
        &texDesc_pos_radius, nullptr));

    // Texture for Sphere Material Index (int) - stored as int
    cudaResourceDesc resDesc_mat_idx = {};
    resDesc_mat_idx.resType = cudaResourceTypeLinear;
    resDesc_mat_idx.res.linear.devPtr = d_spheres_mat_idx;
    resDesc_mat_idx.res.linear.sizeInBytes = num_spheres * sizeof(int);
    resDesc_mat_idx.res.linear.desc = cudaCreateChannelDesc<int>(); // Element type is int
    cudaTextureDesc texDesc_mat_idx = {};
    texDesc_mat_idx.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj_spheres_mat_idx;
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj_spheres_mat_idx, &resDesc_mat_idx,
        &texDesc_mat_idx, nullptr));
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
    render<<<dimGrid, dimBlock>>>(pixel_buffer, cam, d_rand_states,
        texObj_spheres_center_radius, texObj_spheres_mat_idx,
        texObj_materials_albedo_fuzz, texObj_materials_ri, texObj_materials_type);
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
        << "global_float_"
        << "scene" << scene_id
        << "_" << width << "x" << height
        << "_" << samples << "samples"
        << "_" << bounces << "bounces"
        << "_" << threads_per_2d_block_row << "threadsPerBlockRow"
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


    // cudaFree device allocations, destroy textures, delete host heap allocations
    CUDA_SAFE_CALL(cudaFree(pixel_buffer));
    CUDA_SAFE_CALL(cudaFree(d_rand_states));
    CUDA_SAFE_CALL(cudaFree(d_materials_albedo_fuzz));
    CUDA_SAFE_CALL(cudaFree(d_materials_refraction_index));
    CUDA_SAFE_CALL(cudaFree(d_materials_type));
    CUDA_SAFE_CALL(cudaFree(d_spheres_center_radius));
    CUDA_SAFE_CALL(cudaFree(d_spheres_mat_idx));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj_materials_albedo_fuzz));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj_materials_ri));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj_materials_type));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj_spheres_center_radius));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj_spheres_mat_idx));
    delete[] h_materials;
    delete[] h_spheres;

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
