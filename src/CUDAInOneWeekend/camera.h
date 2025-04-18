#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"


class camera {
    public:
        int     img_width;       // rendered image width in pixel count
        int     img_height;      // rendered image height
        point3  center;          // camera center
        point3  pixel00_loc;     // location of pixel 0, 0
        vec3    pixel_delta_u;   // offset to pixel to the right
        vec3    pixel_delta_v;   // offset to pixel below

        camera(int img_width, int img_height) :
            img_width(img_width), img_height(img_height) {initialize();}

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
        }
  };


#endif
