#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "interval.h"


class camera {
    public:
        int     img_width;              // rendered image width in pixel count
        int     img_height;             // rendered image height
        int     samples_per_pixel;      // number of random samples for each pixel
        float   pixel_samples_scale;    // color scale factor for sum of pixel samples
        point3  center;                 // camera center
        point3  pixel00_loc;            // location of pixel 0, 0
        vec3    pixel_delta_u;          // offset to pixel to the right
        vec3    pixel_delta_v;          // offset to pixel below

        camera(int img_width, int img_height, int samples_per_pixel) :
            img_width(img_width), img_height(img_height), samples_per_pixel(samples_per_pixel) {
                initialize();
            }

        void initialize() {
            float focal_length = 1.0f;
            float viewport_height = 2.0f;
            float viewport_width = viewport_height * (float(img_width)/img_height);
            center = point3(0, 0, 0);

            // calculate the vectors across the horizontal and down the vertical viewport edges
            vec3 viewport_u = vec3(viewport_width, 0, 0);
            vec3 viewport_v = vec3(0, -viewport_height, 0);

            // calculate the horizontal and vertical delta vectors from pixel to pixel
            pixel_delta_u = viewport_u / img_width;
            pixel_delta_v = viewport_v / img_height;

            // calculate the location of the upper left pixel
            vec3 viewport_upper_left = center - vec3(0, 0, focal_length)
                                        - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

            pixel_samples_scale = 1.0f / samples_per_pixel;
        }
  };


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

__global__ void render(vec3 *pixel_buffer, camera cam, world *d_world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam.img_width) || (j >= cam.img_height)) return;

    curandState thread_rand_state = rand_state[j*cam.img_width+i];

    color pixel_color(0,0,0);
    int samples_per_pixel = cam.samples_per_pixel;
    int sample = 0;
    for (sample = 0; sample < samples_per_pixel; ++sample) {
        // construct a ray originating from camera center,
        // directed at a randomly sampled point (in a square)
        // around the pixel location i,j
        vec3 offset = vec3(curand_uniform(&thread_rand_state) - 0.5f,
                           curand_uniform(&thread_rand_state) - 0.5f,
                           0);
        point3 pixel_sample = cam.pixel00_loc
                               + ((i + offset.x()) * cam.pixel_delta_u)
                               + ((j + offset.y()) * cam.pixel_delta_v);
        point3 ray_origin = cam.center;
        vec3 ray_direction = pixel_sample - ray_origin;
        ray r(ray_origin, ray_direction);

        // accumulate the various samples' colors
        // NOTE: multiple accumulators and/or loop unrolling here?!!!
        // samples can be incremented by k = 2, 4, 8, etc.
        pixel_color += ray_color(r, *d_world);
    }

    pixel_buffer[j*cam.img_width+i] = cam.pixel_samples_scale * pixel_color;
}


#endif
