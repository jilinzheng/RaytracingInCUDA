
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

// Note: Watch out! Include order will matter!
#include "rtweekend.h"
// #include "camera.h"
// #include "hittable.h"
// #include "hittable_list.h"
// #include "material.h"
// #include "sphere.h"

// assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ float hit_sphere(const point3& center, float radius, const ray& r) {
    vec3 oc = center - r.origin();
    float a = r.direction().length_squared();
    float h = dot(r.direction(), oc);
    float c = oc.length_squared() - radius*radius;
    float discriminant = h*h - a*c;

    if (discriminant < 0) {
        return -1.0f;
    } else {
        return (h - std::sqrt(discriminant)) / a;
    }
}

__device__ color ray_color(const ray& r) {
    float t = hit_sphere(point3(0,0,-1), 0.5f, r);
    if (t > 0.0f) {
        vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
        return 0.5f*color(N.x()+1, N.y()+1, N.z()+1);
    }

    vec3 unit_direction = unit_vector(r.direction());
    float a = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-a)*color(1.0f, 1.0f, 1.0f) + a*color(0.5f, 0.7f, 1.0f);
}

// blue-white gradient from chapter 4
__global__ void render(vec3 *pixel_buffer, int img_width, int img_height,
    point3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, point3 camera_center) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= img_width) || (j >= img_height)) return;

    point3 pixel_center = pixel00_loc+(i*pixel_delta_u)+(j*pixel_delta_v);
    vec3 ray_direction = pixel_center - camera_center;
    ray r(camera_center,ray_direction);
    pixel_buffer[j*img_width+i] = ray_color(r);
}

int main() {
    // select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // 1280 * 800 = 1,024,000 pixels, divisible by warp size 32
    // also divisible by thread block's row size (8)
    int img_width = 640, img_height = 360;      // match serial base
    // int img_width = 1280, img_height = 800;
    int num_pixels = img_width*img_height;

    // buffer to store device-calculated pixels, to later be printed on host;
    // using Unified Memory, i.e., accessible by both host and device
    // float *pixel_buffer;
    vec3 *pixel_buffer;
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&pixel_buffer, num_pixels*sizeof(vec3)));

    // square blocks to start
    int num_threads_per_block_row = 8;
    dim3 dimGrid(img_width/num_threads_per_block_row,
        img_height/num_threads_per_block_row);
    dim3 dimBlock(num_threads_per_block_row,num_threads_per_block_row);

    /* configure the virtual camera */
    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * (float(img_width)/img_height);
    point3 camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    vec3 pixel_delta_u = viewport_u / img_width;
    vec3 pixel_delta_v = viewport_v / img_height;

    // Calculate the location of the upper left pixel.
    vec3 viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    vec3 pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
    /* end virtual camera config */

    // call the render() kernel
    render<<<dimGrid, dimBlock>>>(pixel_buffer, img_width, img_height,
        pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);
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
    CUDA_SAFE_CALL(cudaFree(pixel_buffer));

    return 0;
}
