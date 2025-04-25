#include "rtweekend.h"
#include "hittable.h" // Assuming sphere definition is here
#include "color.h"
#include "camera.h"
#include "curand_kernel.h"
#include "material.h" // Assuming material definition is here
#include <iostream>
#include <iomanip>
#include "cxxopts.hpp"
#include <fstream>
#include <sstream>
#include <filesystem> // For mkdir -p

// assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --- Device Code (Typically in a .cu file) ---

// Define the maximum number of spheres and materials for Scene 1.
// These MUST be compile-time constants for __constant__ memory.
// Based on your scene 1 definition: 1 ground + 22*22 small + 3 big = 1 + 484 + 3 = 488
#define MAX_SCENE1_OBJECTS (1 + 22 * 22 + 3)

// Declare the spheres and materials arrays in constant memory
__constant__ sphere d_spheres_constant[MAX_SCENE1_OBJECTS];
__constant__ material d_materials_constant[MAX_SCENE1_OBJECTS];

// The world struct in constant memory will just need the count,
// as the spheres array is now a global __constant__ array.
// Modify your world struct definition if necessary to remove the sphere* pointer
// if it's not needed when spheres are in a global __constant__ array.
// However, if world struct is small, keeping the pointer might be fine,
// but it would point to the start of the __constant__ array.
// Let's assume the world struct in constant memory just needs the count.
// If your world struct needs more, adjust accordingly.
/*
// Example simplified world struct for constant memory
struct world_constant {
    int num_spheres;
    // Add other fixed-size scene parameters if needed
};
__constant__ world_constant d_world_constant;
*/
// Or, if your original world struct is small and fixed size:
__constant__ world d_world_constant; // Assuming original world struct is small


// The update_world_pointer kernel is no longer needed.
// The update_material_pointers kernel is no longer needed because
// material pointers are set on the host before copying to constant memory.

// Modify the render kernel signature
__global__ void render(vec3 *pixel_buffer, camera cam, curandState *rand_states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= cam.img_width) || (j >= cam.img_height)) return;
    int pixel_index = j * cam.img_width + i;

    curandState thread_rand_state = rand_states[pixel_index];

    color pixel_color(0, 0, 0);
    int samples_per_pixel = cam.samples_per_pixel;
    int sample = 0;
    for (sample = 0; sample < samples_per_pixel; ++sample) {
        // ... (ray construction using cam members and random state) ...
        vec3 offset = vec3(curand_uniform(&thread_rand_state) - 0.5f,
                           curand_uniform(&thread_rand_state) - 0.5f,
                           0);
        point3 pixel_sample = cam.pixel00_loc
                                 + ((i + offset.x()) * cam.pixel_delta_u)
                                 + ((j + offset.y()) * cam.pixel_delta_v);
        point3 ray_origin = (cam.defocus_angle <= 0) ? cam.center
                                                     : defocus_disk_sample(cam, &thread_rand_state);
        vec3 ray_direction = pixel_sample - ray_origin;
        ray r(ray_origin, ray_direction);


        // Access spheres from constant memory array
        // We need to pass the constant spheres array and count to ray_color
        // Or ray_color can access the global __constant__ arrays directly
        // Let's modify ray_color to take the constant arrays and count.
        // This requires changing ray_color's signature.
        // Example:
        // pixel_color += ray_color(r, cam.max_depth, d_spheres_constant, MAX_SCENE1_OBJECTS, d_materials_constant, &thread_rand_state);

        // ALTERNATIVELY, if ray_color is defined in the same .cu file or included
        // after the __constant__ declarations, it can access them directly.
        // Assuming ray_color can see d_spheres_constant and d_materials_constant:
        pixel_color += ray_color(r, cam.max_depth, &thread_rand_state); // Simplified call if ray_color accesses constants directly
    }

    rand_states[pixel_index] = thread_rand_state;

    // ... (color scaling and gamma correction) ...
    pixel_color *= cam.pixel_samples_scale;
    pixel_color = color(linear_to_gamma(pixel_color.x()),
                        linear_to_gamma(pixel_color.y()),
                        linear_to_gamma(pixel_color.z()));
    pixel_buffer[pixel_index] = pixel_color;
}

// Modify ray_color signature and implementation to use constant memory
// Assuming ray_color was something like:
// __device__ color ray_color(const ray& r, int depth, const world& world_data, curandState* rand_state) { ... }
// Now it needs to access the constant arrays.
// Option 1: Pass constant arrays and count
/*
__device__ color ray_color(const ray& r, int depth,
                           const sphere* spheres_const, int num_spheres_const,
                           const material* materials_const,
                           curandState* rand_state) {
    // ... hit logic using spheres_const and num_spheres_const ...
    // When a hit occurs, access material: materials_const[hit_record.mat_idx]
    // You'll need to store material index in hit_record instead of a pointer
}
*/
// Option 2: Access constant arrays directly (if defined in scope)
__device__ color ray_color(const ray& r, int depth, curandState* rand_state) {
    // Access d_spheres_constant and d_materials_constant directly
    // Use MAX_SCENE1_OBJECTS for the count
    // ... hit logic using d_spheres_constant and MAX_SCENE1_OBJECTS ...
    // When a hit occurs, access material: d_materials_constant[hit_record.mat_idx]
    // You'll need to store material index in hit_record instead of a pointer
    hit_record rec;
    // Example hit check (assuming sphere::hit takes constant sphere pointer)
    // for (int i = 0; i < MAX_SCENE1_OBJECTS; ++i) {
    //     if (d_spheres_constant[i].hit(r, interval(0.001, infinity), rec)) {
    //         // ... process hit ...
    //         // Access material: d_materials_constant[rec.mat_idx]
    //     }
    // }

    // Placeholder for ray_color logic using constant memory
    // You need to adapt your actual ray_color implementation
    // to iterate through d_spheres_constant and use d_materials_constant
    color final_color(0,0,0); // Placeholder
    // ... your ray_color logic using d_spheres_constant, MAX_SCENE1_OBJECTS, d_materials_constant ...
    return final_color;
}


// --- Host Code (Typically in a .cpp file) ---

// Declare the constant variables (matches device declarations)
__constant__ sphere d_spheres_constant[MAX_SCENE1_OBJECTS];
__constant__ material d_materials_constant[MAX_SCENE1_OBJECTS];
__constant__ world d_world_constant; // Assuming original world struct is small

int main(int argc, char* argv[]) {
    // ... (parsing, device selection, timing setup, camera setup) ...

    /* world creation as determined by scene_id */
    // host allocations and initializations
    int num_materials, num_spheres;
    material *h_materials;
    sphere *h_spheres;

    switch (scene_id) {
        // original scene from end of book
        case 1: {
            // For constant memory, the size is fixed at compile time.
            // We must use the defined constant size.
            num_materials = MAX_SCENE1_OBJECTS; // Should be 488
            num_spheres = MAX_SCENE1_OBJECTS;   // Should be 488

            // Allocate host memory for building the scene
            h_materials = new material[num_materials];
            h_spheres = new sphere[num_spheres];

            // ground sphere (index 0)
            h_materials[0] = material(MaterialType::LAMBERTIAN, color(0.5,0.5,0.5));
            // IMPORTANT: For constant memory, sphere struct cannot contain a pointer
            // to material that needs device-side update. It should store a material INDEX.
            // Modify your sphere struct to use material index instead of pointer.
            // Example: sphere(point3 center, double radius, int material_idx);
            // Then h_spheres[0] would be sphere(point3(0,-1000,0), 1000, 0); // Index 0 for ground material

            // --- Re-evaluate your sphere and material struct definitions ---
            // If sphere struct has 'material* mat', you CANNOT put it in constant memory
            // and update 'mat' pointers on the device. The pointers must be valid
            // device pointers *before* copying to constant memory.
            // Option A (Requires sphere struct change): sphere stores material index (int)
            // Option B (More complex): sphere stores material pointer, but you need
            // to calculate the device address of the material in d_materials_constant
            // on the host and set the pointer before copying the sphere to constant memory.
            // Option A is generally cleaner for constant memory.

            // Assuming sphere struct is modified to use material_idx (int)
            // h_spheres[0] = sphere(point3(0,-1000,0), 1000, 0); // Use material index 0

            // If sphere struct *must* contain material* mat, and you want spheres in constant memory:
            // You need to calculate the device address of the material in d_materials_constant
            // on the host side before copying the sphere to d_spheres_constant.
            // This is tricky because you need the device address of d_materials_constant on the host.
            // You can get the address of a __constant__ symbol on the host like this:
            // material* d_materials_base_address;
            // cudaGetSymbolAddress((void**)&d_materials_base_address, d_materials_constant);
            // Then, for each sphere: h_spheres[i].mat = d_materials_base_address + h_spheres[i].material_index_from_host;
            // This is error-prone. Storing material index in sphere is better for constant memory.

            // LET'S ASSUME sphere struct is modified to use material_idx (int)
            // You will need to update your sphere struct definition.
            // Example: struct sphere { point3 center; double radius; int material_idx; };
            // And material struct might need an ID or index if you have multiple types.

            // --- Let's proceed assuming sphere struct uses material_idx ---
            // You will need to populate h_materials and h_spheres with material indices.
            // Example for ground sphere:
            h_materials[0] = material(MaterialType::LAMBERTIAN, color(0.5,0.5,0.5)); // Material at host index 0
            h_spheres[0] = sphere(point3(0,-1000,0), 1000, 0); // Sphere uses material at index 0

            // small spheres (using the corrected indexing logic from before)
            int current_sphere_idx = 1; // Start after ground sphere
            for (int a = -11; a < 11; ++a) {
                for (int b = -11; b < 11; ++b) {
                    float choose_mat = random_float();
                    point3 center(a+0.9*random_float(), 0.2, b+0.9*random_float());

                    if ((center - point3(4,0.2,0)).length() > 0.9) {
                         // Use the counter for sequential assignment
                         int h_idx = current_sphere_idx; // Host index for this sphere/material

                         // Ensure we don't exceed the allocated size for small spheres (up to index 1 + 22*22 - 1 = 484)
                         if (h_idx < (1 + 22 * 22)) {
                            // Assign material properties to h_materials[h_idx]
                            // Assign sphere properties to h_spheres[h_idx] and set its material_idx to h_idx

                            // diffuse
                            if (choose_mat < 0.8) {
                                color albedo    = color::random() * color::random();
                                h_materials[h_idx]  = material(MaterialType::LAMBERTIAN, albedo);
                                h_spheres[h_idx]    = sphere(center, 0.2, h_idx); // Use material index h_idx
                            }
                            // metal
                            else if (choose_mat < 0.95) {
                                color albedo    = color::random(0.5,1.0);
                                float fuzz      = random_float(0.0,0.5);
                                h_materials[h_idx]  = material(MaterialType::METAL, albedo, fuzz);
                                h_spheres[h_idx]    = sphere(center, 0.2, h_idx); // Use material index h_idx
                            }
                            // glass
                            else {
                                h_materials[h_idx]  = material(MaterialType::DIELETRIC, 1.5);
                                h_spheres[h_idx]    = sphere(center, 0.2, h_idx); // Use material index h_idx
                            }
                            current_sphere_idx++; // Increment counter only when a sphere is added
                         }
                    }
                }
            }

            // shared 3 big spheres, start index after small spheres
            // These should start at index 1 + (number of small spheres created)
            // If the condition always passes, this is 1 + 484 = 485.
            // If the condition can skip, this is 1 + current_sphere_idx (after the loop).
            // Let's assume the 22x22 grid always creates 484 spheres for simplicity of constant size.
            // If not, you need a different strategy or a fixed size array that might have empty slots.
            // Assuming 484 small spheres are always intended for scene 1:
            int big_sphere_start_index = 1 + (22 * 22); // Index 485

            // middle sphere (index 485)
            h_materials[big_sphere_start_index] = material(MaterialType::DIELETRIC, 1.5);
            h_spheres[big_sphere_start_index]   = sphere(point3(0,1,0), 1.0, big_sphere_start_index); // Use material index
            // rear sphere (index 486)
            h_materials[big_sphere_start_index+1] = material(MaterialType::LAMBERTIAN, color(0.4,0.2,0.1));
            h_spheres[big_sphere_start_index+1]   = sphere(point3(-4,1,0), 1.0, big_sphere_start_index+1); // Use material index
            // front sphere (index 487)
            h_materials[big_sphere_start_index+2] = material(MaterialType::METAL, color(0.7,0.6,0.5), 0.0);
            h_spheres[big_sphere_start_index+2]   = sphere(point3(4,1,0), 1.0, big_sphere_start_index+2); // Use material index

            // At this point, h_spheres and h_materials are fully populated for scene 1.
            // num_spheres and num_materials should be MAX_SCENE1_OBJECTS (488).

            break;
        }
        case 2: {
             // Your existing case 2 code (if you are not putting these in constant memory)
             // If you want these in constant memory too, you need separate __constant__ arrays
             // and a separate MAX_SCENE2_OBJECTS constant.
             num_materials = 1+6*6+3;
             num_spheres = num_materials;
             h_materials = new material[num_materials]; // These will be allocated on host heap
             h_spheres = new sphere[num_spheres];       // These will be allocated on host heap
             // ... populate h_materials and h_spheres for case 2 ...
             // If not using constant memory for case 2, you'll need to cudaMalloc/cudaMemcpy
             // d_spheres and d_materials for case 2, and pass them to the kernel.
             // This makes the kernel signature scene-dependent or requires a different kernel.
             // For simplicity of this example, we focus only on putting Scene 1 into constant memory.
             break;
        }
        default: {
            // Your existing default case code (similar considerations as case 2)
            num_materials = 1+11*11+3;
            num_spheres = num_materials;
            h_materials = new material[num_materials]; // Host heap
            h_spheres = new sphere[num_spheres];       // Host heap
            // ... populate h_materials and h_spheres for default case ...
            // cudaMalloc/cudaMemcpy for these if not using constant memory
            break;
        }
    } // End of switch

    // Create the host-side world object *after* h_spheres is populated
    // This h_world object is just a temporary struct to copy to constant memory.
    // It should contain the count of spheres for the scene being rendered.
    // If rendering scene 1 (constant memory), use MAX_SCENE1_OBJECTS.
    // If rendering other scenes (global memory), use their num_spheres.
    world h_world; // Assuming world struct is small and fixed size
    if (scene_id == 1) {
        h_world.num_spheres = MAX_SCENE1_OBJECTS;
        // If world struct needs a pointer to spheres, it should point to the host array for copying
        h_world.spheres = h_spheres; // Point to the host array for copying
    } else {
        // For other scenes, h_world needs to be populated based on their num_spheres
        h_world.num_spheres = num_spheres; // Use the dynamically determined count
        h_world.spheres = h_spheres; // Point to the host array
    }


    // --- Copy data to __constant__ memory on the device ---
    // This replaces cudaMalloc/cudaMemcpy for d_spheres, d_materials, and d_world
    // This copy is ONLY done if rendering scene 1.
    if (scene_id == 1) {
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_spheres_constant, h_spheres, num_spheres * sizeof(sphere)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_materials_constant, h_materials, num_materials * sizeof(material)));
        // Copy the world struct itself to constant memory
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_world_constant, &h_world, sizeof(world)));

        // Free the host arrays AFTER copying to constant memory
        delete[] h_materials;
        delete[] h_spheres;
        // delete h_world; // h_world is a stack variable here if declared inside if/else
                         // If declared outside, delete it after copying.
    } else {
        // --- For other scenes (not in constant memory) ---
        // You still need to allocate and copy to global memory
        material *d_materials;
        CUDA_SAFE_CALL(cudaMalloc(&d_materials, num_materials * sizeof(material)));
        CUDA_SAFE_CALL(cudaMemcpy(d_materials, h_materials, num_materials * sizeof(material),
                                 cudaMemcpyHostToDevice));

        sphere *d_spheres;
        CUDA_SAFE_CALL(cudaMalloc(&d_spheres, num_spheres * sizeof(sphere)));
        CUDA_SAFE_CALL(cudaMemcpy(d_spheres, h_spheres, num_spheres * sizeof(sphere),
                                 cudaMemcpyHostToDevice));

        world *d_world; // Allocate world struct in global memory
        CUDA_SAFE_CALL(cudaMalloc(&d_world, sizeof(world)));
        CUDA_SAFE_CALL(cudaMemcpy(d_world, &h_world, sizeof(world),
                                 cudaMemcpyHostToDevice));

        // Update material pointers on the device for global memory arrays
        update_material_pointers<<<1, 1>>>(d_spheres, d_materials, num_spheres);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // Update world pointer for global memory world struct
        update_world_pointer<<<1, 1>>>(d_world, d_spheres);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // Free host arrays after copying to global memory
        delete[] h_materials;
        delete[] h_spheres;
        // delete h_world; // If h_world was allocated on heap
    }


    /* end world creation*/


    // setup random number generation in device
    curandState *d_rand_states;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rand_states, num_pixels * sizeof(curandState)));
    init_rng<<<dimGrid, dimBlock>>>(cam.img_width, cam.img_height, d_rand_states);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // call the render() kernel
    // start (render) kernel-only timing
    cudaEventRecord(render_only_start, 0);

    // Launch render kernel. The arguments depend on where the world data is.
    if (scene_id == 1) {
         // For scene 1, world data is in constant memory, no pointer needed
         render<<<dimGrid, dimBlock>>>(pixel_buffer, cam, d_rand_states);
    } else {
         // For other scenes, world data is in global memory, pass the pointer
         // You would need a d_world pointer allocated in the else block
         // render<<<dimGrid, dimBlock>>>(pixel_buffer, cam, d_world, d_rand_states); // Need d_world pointer here
         // This means your render kernel signature might need to change back to
         // accept a world pointer, and inside the kernel, you check if it's null
         // or use a different kernel for constant vs global memory.
         // A simpler approach might be to have two different render kernels or
         // use template parameters if the world access pattern is very different.
         // For this example, we assume the render kernel is specifically for constant memory access.
         // If you need to handle both, the render kernel signature needs the world* parameter back,
         // and you'd pass d_world for global scenes, and perhaps NULL or a dummy pointer for constant scenes.
         // Inside the kernel, you'd check the pointer or use if/else based on a scene ID passed to the kernel.

         // Let's assume for this refactoring, the render kernel is specifically
         // designed to access the __constant__ memory for scene 1.
         // If you need other scenes, you might need a different kernel or approach.
         std::cerr << "Error: Only scene_id 1 is implemented with constant memory." << std::endl;
         return 1;
    }

    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // stop (render) kernel-only timing
    cudaEventRecord(render_only_stop, 0);
    cudaEventSynchronize(render_only_stop);
    cudaEventElapsedTime(&render_only_elapsed, render_only_start, render_only_stop);
    std::cout << std::fixed << std::setprecision(8)
              << std::setw(15) << render_only_elapsed << ",";
    cudaEventDestroy(render_only_start);
    cudaEventDestroy(render_only_stop);

    /* begin writing pixel_buffer out to .ppm file */
    // ... (your file writing code using ofs) ...
    std::stringstream f_ss;
    f_ss
        << "scene" << scene_id
        << "_" << width << "x" << height
        << "_" << samples << "samples"
        << "_" << bounces << "bounces"
        << "_" << threads_per_2d_block_row << "threadsPerBlockRow"
        << ".ppm";
    std::string f = f_ss.str();

    std::ofstream ofs(f);
    if (!ofs) {
        std::cerr << "Error: Could not open file for writing: " << f << "\n";
        return -1;
    }

    const interval intensity(0.000, 0.999);
    ofs << "P3\n" << cam.img_width << " " << cam.img_height << "\n255\n";
    for (int j = 0; j < cam.img_height; ++j) {
        for (int i = 0; i < cam.img_width; ++i) {
            size_t pixel_index = j * cam.img_width + i;
            vec3 pixel = pixel_buffer[pixel_index];
            int r = static_cast<int>(256 * intensity.clamp(pixel.x()));
            int g = static_cast<int>(256 * intensity.clamp(pixel.y()));
            int b = static_cast<int>(256 * intensity.clamp(pixel.z()));
            ofs << r << " " << g << " " << b << "\n";
        }
    }
    /* end writing */


    // cudaFree device allocations, delete host heap allocations
    CUDA_SAFE_CALL(cudaFree(pixel_buffer));
    CUDA_SAFE_CALL(cudaFree(d_rand_states));

    // Only free d_materials, d_spheres, d_world if they were allocated in global memory (for non-scene 1)
    if (scene_id != 1) {
        // You need to keep track of d_materials, d_spheres, d_world pointers in the else block
        // CUDA_SAFE_CALL(cudaFree(d_materials)); // Need pointer from else block
        // CUDA_SAFE_CALL(cudaFree(d_spheres));   // Need pointer from else block
        // CUDA_SAFE_CALL(cudaFree(d_world));     // Need pointer from else block
    }

    // Only delete host heap allocations if they were created (for non-scene 1)
     if (scene_id != 1) {
        // delete[] h_materials; // Need pointer from else block
        // delete[] h_spheres;   // Need pointer from else block
        // delete h_world;      // Need pointer from else block
     } else {
        // h_materials and h_spheres were deleted after copying to constant memory
        // h_world was a stack variable
     }


    // ... (timing teardown and return) ...
    cudaEventRecord(end_to_end_stop, 0);
    cudaEventSynchronize(end_to_end_stop);
    cudaEventElapsedTime(&end_to_end_elapsed, end_to_end_start, end_to_end_stop);
    std::cout << std::fixed << std::setprecision(8)
              << std::setw(15) << end_to_end_elapsed << "\n";
    cudaEventDestroy(end_to_end_start);
    cudaEventDestroy(end_to_end_stop);


    return 0;
}
