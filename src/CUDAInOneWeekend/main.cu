
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
#include "camera_cuda.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

// assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void render(float *pixel_buffer, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x*3 + i*3;
    pixel_buffer[pixel_index + 0] = float(i) / max_x;
    pixel_buffer[pixel_index + 1] = float(j) / max_y;
    pixel_buffer[pixel_index + 2] = 0.2;
}

// world generation
int main() {
    // select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // 1280 * 800 = 1,024,000 pixels, divisible by warp size 32
    // also divisible by thread block's row size (8)
    int img_width = 1280, img_height = 800;
    int num_pixels = img_width*img_height;
    // square blocks to start
    int num_threads_per_block_row = 8;
    dim3 dimGrid(img_width/num_threads_per_block_row,
        img_height/num_threads_per_block_row);
    dim3 dimBlock(num_threads_per_block_row,num_threads_per_block_row);

    // buffer to store device-calculated pixels, to later be printed on host;
    // using Unified Memory, i.e., accessible by both host and device
    vec3 *pixel_buffer;
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&pixel_buffer, num_pixels*sizeof(vec3)));

    // call the render() kernel
    render<<<dimGrid, dimBlock>>>(pixel_buffer, img_width, img_height);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    return 0;
}
