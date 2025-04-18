// Note: Watch out! Include order will matter!
#include "rtweekend.h"
#include "hittable.h"
#include "color.h"
#include "camera.h"
// #include "material.h"

// assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ color ray_color(const ray& r, const world& world) {
    // track the hits for this particular ray
    hit_record rec;

    // hits will be the sphere's surface normal
    if (hit_world(world, r, interval(0.0f, infinity), rec)) {
        return 0.5f * (rec.normal + color(1,1,1));
    }

    // background blue-to-white gradient
    vec3 unit_direction = unit_vector(r.direction());
    float a = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-a)*color(1.0f, 1.0f, 1.0f) + a*color(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3 *pixel_buffer, camera cam, world *d_world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam.img_width) || (j >= cam.img_height)) return;

    point3 pixel_center = cam.pixel00_loc+(i*cam.pixel_delta_u)+(j*cam.pixel_delta_v);
    vec3 ray_direction = pixel_center - cam.center;
    ray r(cam.center,ray_direction);
    pixel_buffer[j*cam.img_width+i] = ray_color(r,*d_world);
}

__global__ void update_world_pointer(world *w, sphere *spheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        w->spheres = spheres;
    }
}

int main() {
    // select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // 1280 * 800 = 1,024,000 pixels, divisible by warp size 32
    // also divisible by thread block's row size (8)
    // these dimensions match serial base
    int img_width = 640, img_height = 360;
    // these dimensions match the CUDA reference
    // int img_width = 1280, img_height = 800;
    int num_pixels = img_width*img_height;

    // buffer to store device-calculated pixels, to later be printed on host;
    // using Unified Memory, i.e., accessible by both host and device
    vec3 *pixel_buffer;
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&pixel_buffer, num_pixels*sizeof(vec3)));

    // square blocks to start
    int num_threads_per_block_row = 8;
    dim3 dimGrid(img_width/num_threads_per_block_row,
        img_height/num_threads_per_block_row);
    dim3 dimBlock(num_threads_per_block_row,num_threads_per_block_row);

    // initialize the camera
    camera cam(img_width,img_height);

    /* world creation */
    // host allocations and initializations
    int num_spheres = 2;
    sphere *h_spheres = new sphere[num_spheres];
    h_spheres[0] = sphere(point3(0,0,-1), 0.5f);
    h_spheres[1] = sphere(point3(0,-100.5f,-1), 100);

    world *h_world = new world(h_spheres,num_spheres);

    // device allocations and transfers
    sphere *d_spheres;
    cudaMalloc(&d_spheres, num_spheres*sizeof(sphere));
    cudaMemcpy(d_spheres,h_spheres,num_spheres*sizeof(sphere),cudaMemcpyHostToDevice);

    world *d_world;
    cudaMalloc(&d_world,sizeof(world));
    cudaMemcpy(d_world,h_world,sizeof(world),cudaMemcpyHostToDevice);

    // update needed since the host pointer to spheres will be invalid
    // after transferring to device
    update_world_pointer<<<1,1>>>(d_world, d_spheres);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    /* end world creation*/

    // call the render() kernel
    render<<<dimGrid, dimBlock>>>(pixel_buffer, cam, d_world);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // output pixel_buffer as a .ppm image
    std::cout << "P3\n" << img_width << " " << img_height << "\n255\n";
    for (int j = 0; j < img_height; ++j) {      // rows
        for (int i = 0; i < img_width; ++i) {   // cols
            size_t pixel_index = j*img_width+i;
            int ir = int(255.99f*pixel_buffer[pixel_index].x());
            int ig = int(255.99f*pixel_buffer[pixel_index].y());
            int ib = int(255.99f*pixel_buffer[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // cudaFree and delete everything
    CUDA_SAFE_CALL(cudaFree(pixel_buffer));
    delete h_spheres;
    delete h_world;

    return 0;
}
