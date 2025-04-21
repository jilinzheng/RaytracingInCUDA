#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "interval.h"
#include "material.h"
#include "color.h"


struct camera {
    int     img_width;              // rendered image width in pixel count
    int     img_height;             // rendered image height
    int     samples_per_pixel;      // number of random samples for each pixel
    float   pixel_samples_scale;    // color scale factor for sum of pixel samples
    int     max_depth;              // maximum recursion depth for ray bounces
    point3  center;                 // camera center
    point3  pixel00_loc;            // location of pixel 0, 0
    vec3    pixel_delta_u;          // offset to pixel to the right
    vec3    pixel_delta_v;          // offset to pixel below

    float   vfov     = 90;              // vertical view angel (field of view)
    point3  lookfrom = point3(0,0,0);   // point camera is looking from
    point3  lookat   = point3(0,0,0);   // point camera is looking from
    vec3    vup      = vec3(0,1,0);     // camera-relative "up" direction
    vec3    u,v,w;                      // camera frame basis vectors

    float   defocus_angle = 0;  // variation angle of rays through each pixel
    float   focus_dist = 10;    // distance from camera lookfrom point to plane of perfect focus
    vec3    defocus_disk_u;     // defocus disk horizontal radius
    vec3    defocus_disk_v;     // defocus disk vertical radius


    void initialize() {
        pixel_samples_scale = 1.0f / samples_per_pixel;

        center = lookfrom;

        // NOTE: since camera initialization is done in host, we could
        // potentially go back to double-precision for more, well, precision
        // but not sure how exactly it will affect the device kernels...
        // float focal_length = (lookfrom - lookat).length();
        float theta = degrees_to_radians(vfov);
        float h = std::tan(theta/2);
        float viewport_height = 2.0f * h * focus_dist;
        float viewport_width = viewport_height * (float(img_width)/img_height);

        // calculate u,v,w unit basis vectors for camera coordinate frame
        w = unit_vector(lookfrom-lookat);
        u = unit_vector(cross(vup,w));
        v = cross(w,u);

        // calculate vectors across horizontal and down vertical viewport edges
        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        // calculate horizontal and vertical delta vectors from pixel to pixel
        pixel_delta_u = viewport_u / img_width;
        pixel_delta_v = viewport_v / img_height;

        // calculate location of the upper left pixel
        vec3 viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        // calculate camera defocus disk basis vectors
        float defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle/2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }


  };

__device__ point3 defocus_disk_sample(camera& cam,curandState *thread_rand_state) {
    point3 p = random_in_unit_disk(thread_rand_state);
    return cam.center + (p[0] * cam.defocus_disk_u) + (p[1] * cam.defocus_disk_v);
}

__device__ color ray_color(const ray& r, int max_depth, const world& world,
    curandState *thread_rand_state) {
    // for loop instead of recursion;
    // GPU freaks out when not being able to detect stack size...
    ray curr_ray = r;
    color curr_attenuation = color(1.0f,1.0f,1.0f);
    for (int i = 0; i < max_depth; ++i) {
        // track hits
        hit_record rec;
        if (hit_world(world, curr_ray, interval(0.001f,infinity), rec)) {
            ray scattered;
            color attenuation;
            switch (rec.mat->type) {
                case MaterialType::LAMBERTIAN: {
                    if (lambertian_scatter(r,rec,attenuation,scattered,thread_rand_state)) {
                        curr_attenuation = curr_attenuation * attenuation;
                        curr_ray = scattered;
                        break;
                    }
                    else return color(0.0f,0.0f,0.0f);
                }
                case MaterialType::METAL: {
                    if (metal_scatter(r,rec,attenuation,scattered,thread_rand_state)) {
                        curr_attenuation = curr_attenuation * attenuation;
                        curr_ray = scattered;
                        break;
                    }
                    else return color(0.0f,0.0f,0.0f);
                }
                case MaterialType::DIELETRIC: {
                    if (dieletric_scatter(r,rec,attenuation,scattered,thread_rand_state)) {
                        curr_attenuation = curr_attenuation * attenuation;
                        curr_ray = scattered;
                        break;
                    }
                    else return color(0.0f,0.0f,0.0f);
                }
            }
        }
        // blue-to-white gradient background
        else {
            vec3 unit_direction = unit_vector(r.direction());
            float a = 0.5f * (unit_direction.y() + 1.0f);
            return curr_attenuation*((1.0f-a)*color(1.0f,1.0f,1.0f)+a*color(0.5f,0.7f,1.0f));
        }
    }
    // max depth reached
    return color(0,0,0);
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
        // around the pixel location i,j; NOTE: this is get_ray(i,j)
        vec3 offset = vec3(curand_uniform(&thread_rand_state) - 0.5f,
                           curand_uniform(&thread_rand_state) - 0.5f,
                           0);
        point3 pixel_sample = cam.pixel00_loc
                               + ((i + offset.x()) * cam.pixel_delta_u)
                               + ((j + offset.y()) * cam.pixel_delta_v);
        // point3 ray_origin = cam.center;
        point3 ray_origin = (cam.defocus_angle<=0) ? cam.center
                                                    : defocus_disk_sample(cam,&thread_rand_state);
        vec3 ray_direction = pixel_sample - ray_origin;
        ray r(ray_origin, ray_direction);

        // accumulate the various samples' colors
        // NOTE: multiple accumulators and/or loop unrolling here?!!!
        // samples can be incremented by k = 2, 4, 8, etc.
        pixel_color += ray_color(r, cam.max_depth, *d_world, &thread_rand_state);
    }

    // scale color back down (divide by number of samples)
    // and perform gamma correction
    pixel_color *= cam.pixel_samples_scale;
    pixel_color = color(linear_to_gamma(pixel_color.x()),
                        linear_to_gamma(pixel_color.y()),
                        linear_to_gamma(pixel_color.z()));
    pixel_buffer[j*cam.img_width+i] = pixel_color;
}


#endif
