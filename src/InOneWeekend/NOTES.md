# Notes for Ray Tracing in One Weekend

## 1 Overview

- most production movie and video game renderers are written in C++
- inheritance and operator overloading are very useful for ray tracers
- style is pretty C-like
- _endless ways one can optimize and modernize the code_

## 2 Output an Image

- PPM image format; write out to cout but redirect to image file
- progress indicator using clog (logging output stream)

## 3 The vec3 Class

- use same vec3 class for colors, locations, directions, offsets, etc.
- `vec3:Point3` and `vec3:color` aliases
- `vec3.h` has the vec3 class and a set of useful vector utility functions
- using `double` while some ray tracers can use `float`; important in limited memory conditions
- `color.h` uses `vec3` and writes a single pixel's color out to the standard output stream cout

## 4 The ray Class

### 4.1 The ray Class

- a ray is function P(t) = A + t\*b, where P is a 3D position along a line in 3D, A is the ray origin, and b is the ray direction
- ray parameter t is a real number (`double` in the code)
- for positive t you get only the parts in front of A, and this is what is often called a half-line/ray
- `ray::at(t)` is the implementation of function P(t)
- `ray::origin()` and `ray::direction()` return immutable references to their members (`orig` and `dir`)

### 4.2 Sending Rays Into the Scene

- at its core, a ray tracer sends rays through pixels and computes the color seen in the direction of those rays; the steps are:
  1. calculate the ray from the "eye" (camera) through the pixel
  2. determine which objects the ray intersects, and
  3. compute a color for the closest intersection point
- use image width to calculate height to keep aspect ratio consistentxdfff- viewport: virtuaxfl rectangle in 3D world that contains the grid of image pixel locations
  - if pixels are spaced the same distance horizontally as they are vertically, the viewport that bounds them will have the same aspect ratio as the rendered image
  - the distance between two adjacent pixels is called the pixel spacing, and square pixels is the standard
- since `aspect_ratio` is an ideal ratio, it may not be the _actual_ ratio between `image_width` and `image_height`, as `image_height` is rounded down to the nearest integer and it cannot be less than one
  - in order for viewport proportions to match image proportions, we use the _calculated_ image aspect ratio to determine our final viewport width
- camera center: a point in 3D space from which all scene rays will originate (a.k.a. eye point)
  - vector from camera center to the viewport center will be orthogonal to the viewport
  - focal length: distance between the camera center and the viewport
  - start with camera center at origin (0,0,0)
  - y-axis goes up, x-axis to the right, and negative z-axis points in viewing direction; right-handed coordinates
- camera center coordinates conflict with image coordinates, where zeroth pixel is the top-left and last pixel is bottom right; image coordinate y-axis is inverted, increasing while going down the image
  - use a vector from left edge to right edge V_u (`viewport_u`) and vector from upper edge to lower edge V_v (`viewport_v`)
- pixel grid inset from viewport edges by half the pixel-to-pixel distance
  - as a result viewport area is evenly divided into width x height identical regions
- camera setup code in `camera.h`
- linear blend = linear interpolation = lerp
  - blendedValue = (1 - a) \* startValue + a \* endValue, where a goes from 0 to 1

## 5 Adding a Sphere

- people often use spheres in ray tracers because calculating whether a ray hits a sphere is relatively simple

## 5.1 Ray-Sphere Intersection

- the equation of a sphere of radius r centered at the origin:
  - x^2 + y^2 + z^2 = r^2
  - if a given point (x,y,z) is on the surface, then the equation is satisfied
  - if a given point is inside, x^2 + y^2 + z^2 < r^2
  - if a given point is outside, x^2 + y^2 + z^2 > r^2
- given point P = (x,y,z), the vector from P to center C = (C_x,C_y,C_z) is (C-P)
  - rewrite the sphere equation as (C-P) . (C-P) = r^2
  - some vector algebra resolves into this into a quadratic equation, and further yields that the discriminant can be interpreted for if a sphere intersects:
    - if discriminant is positive, 2 real solutions, 2 points of interesction
    - if discriminat is zero, 1 real solution, 1 point of interesction
    - if discriminant is negative, no real solutions, no points of intersection

## 5.2 Creating Our First Raytraced Image

- we test if a ray intersects with the sphere by solving the quadratic equation for t
- though currently, negative values of t also work! this current implementation does not distinguish between objects in front of the camera and objects behind the camera

## 6 Surface Normals and Multiple Objects

### 6.1 Shading with Surface Normals

- surface normal: vector that is perpendicular to the surface at the point of intersection
- key design decision: whether normal vectors will have an arbitrary length or will be normalized to unit length
  - it is tempting to skip the expensive square root, but it is needed in some places, so you might as well do it up front once if you need it
  - more importantly though, you can generate a vector with an understanding of the specific geometry class; for example, sphere normals can be made unit length simply by dividing by the sphere radius, avoiding the square root entirely
    - this is the `outward_normal` variable in the `hit()` function in the `sphere` class (`sphere.h`)
  - the book adopts the policy that all normal vectors will be of unit length
- for a sphere, the outward normal is in the direction of the hit point minus the center

### 6.2 Simplifying the Ray-Sphere Intersection Code

- nothing crazy here, just rewriting `hit_sphere` (later named `hit()` function in the `sphere` class (`sphere.h`))

### 6.3 An Abstraction for Hittable Objects

- "abstract class" for anything a ray might hit: `hittable`
  - could be a sphere, list of spheres, fog, clouds, etc.
- `hit` function that takes in a ray
- most ray tracers have a valid interval for hits t_min to t_max so the hit only "counts" if t_min < t < t_max
- `hit_record` stores "a bundle of stuff"...

### 6.4 Front Faces Versus Back Faces

- need to choose if the normal always points out of the sphere, or if the normal always points against the direction of the ray
  - book chooses always...what? can't understand - something about choosing at time of geometry intersection

### 6.5 A List of Hittable Objects

- new class that stores a list of hittables

### 6.6 Some New C++ Features

- `shared_ptr<type>` is a pointer to some allocated type, with reference-counting semantics; once reference count goes to zero, the object is safely deleted
  - `make_shared<thing>(thing_constructor_params...)` allocates a new instance of type `thing`, using the constructor parameters; it returns a `shared_ptr<thing>`
  - `std::shared_ptr` is included with the `<memory>` header

### 6.7 Common Constants and Utility Functions

- in `rtweekend.h`
- big reorganization of main and header files

### 6.8 An Interval Class

- manage real-valued intervals with a minimmum and a maximum
- used throughout several header files

## 7 Moving Camera Code Into Its Own Class

- `camera` class: responsible for two important jobs:
  1. construct and dispatch rays into the world
  2. use the results of these rays to construct the rendered image
- two public methods: `initialize()` and `render()`, two private helper methods `get_ray()` and `ray_color()`
- calls `initialize()` at the start of `render()`

## 8 Antialiasing

- jaggedness/"stair step" edges in rendered images is commonly referred to as "aliasing"
  - when a real camera takes a picture, there is no aliasing because the edges are a blend of foreground and background; it also has effectively infinite resolution
    - we can achieve this by averaging a bunch of samples per pixel
- point sampling (single ray through the center of each pixel) *is problematic* because it does *not* integrate the light falling on a discrete region of an image
  - we **want** our ray tracer to integrate the continuous light falling on a dsicrete region of a rendered image and **avoid** point sampling
  - we sample the **square region** centered at the target pixel that extends halfway to each of the four neighboring pixels

### 8.1 Some Random Number Utilities

- found in `rtweekend.h`; random numbers fall in the range [0,1)

### 8.2 Generating Pixels with Multiple Samples

- for a single pixel composed of multiple samples, we select sampels from the area surrounding the pixel and average the resulting light/color values together
  - add full colors from each sample, then divide by the number of total samples, before writing the color
    - `interval::clamp` ensures final result remains within the proper [0,1) bounds (used in `color.h`)
- in `camera.h`, `camera::get_ray(i,j)` generates different samples for each pixel using the `sample_square()` helper function that generates a random sample point within the unit square centered at the origin
  - this is then transformed from the ideal square back to the particular pixel currently being sampled
  - `sample_disk()` is an alternative method used in a future book; it also relies on `random_in_unit_disk()`, defined later on

## 9 Diffuse Materials

- diffuse = matte
- there are multiple approaches, but this book separates geometry and materials (so that a material can be assigned to multiple sphere/vice-versa)

### 9.1 A Simple Diffuse Material

- diffuse/matte objects that don't emit their own light take on colors of the environment, but do modulate the environment colors with their own intrinsic colors
- light that reflects off a diffuse surface has its direction randomized
- light can also be absorbed instead of reflected; the darker the surface, the more likely the ray is absorbed
- the first algorithm used for diffuse materials here has an *equal* probability for a ray to bounce in any direction away from the surface
  - some more random vector utility functions are in `vec3.h`
- to manipulate random vectors so that the resulting vectors are on the surface of a hemisphere, a **rejection method** is used
  - a rejection method works by repeatedly generating random samples until we produce a sample that meets the desired criteria, i.e., keep rejecting bad samples until you find a good one
- the rejection method here is as follows:
  1. generate a random vector inside the unit sphere
  2. normalize this vector to extend it to the sphere surface
  3. invert the normalized vecotor if it falls onto the wrong hemisphere
- `random_unit_vector()` is in `vec3.h`
  - also rejects the "black hole" around the center, due to a floating-point underflow issue where coordinates close to the center of the sphere (close to zero) can make the norm of the vector zero, and cause the normalization to yield infinity as coordinates of the vector
- as for determining if it is in the right hemisphere, we can use the dot product between the surface normal and the random vector:
  - if the dot product is positive, then the random vector is already in the right hemisphere
  - if the dot product is negative, then we invert the vector to be in the right hemisphere
  - `vec3.h` has a `random_on_hemisphere()` function
- if a ray bounces off of a material and keeps
  - 100% of its color, we say the material is *white*
  - 0% of its color, we say the material is *black*

### 9.2 Limiting the Number of Child Rays

- `ray_color()` is recursive and stops when it fails to hit anything; to prevent the stack from overflowing, a `max_depth` is used, where no light is returned at maximum depth

### 9.3 Fixing Shadow Acne

- a ray will attempt to accurately calculate the intersection point when it intersects with a surface; however, this calculation is susceptible to floating-point rounding errors which can cause the intersection point to be slightly off
  - the simplest fix is just to ignore hits that are very close to the calculated intersection point

### 9.4 True Lambertian Reflection

- a more accurate representation of diffuse objects' scattering is the non-uniform Lambertian distribution, which scatters reflected rays proportional to cos(phi), where phi is the angle between the reflected ray and the surface normal (reminder that previously it was equally scatted inn all directions)
  - this results in a higher probability of the reflected ray scattering in a direction near the surface normal, and less likley to scatter in directions away from the normal
- the implementation becomes `hit-record`'s normal + `random_unit_vector()` in `ray_color()`

### 9.5 Using Gamma Correction for Accurate Color Intensity

- up until now, images have been stored in linear space instead of gamma space (the transformation has not been applied), while computer programs to view images expect that images have been "gamma corrected" (in gamma space)
  - transform in `color.h`

## 10 Metal

### 10.1 An Abstract Class for Materials

- the author likes material classes that encapsulate each material's unique behavior, instead of an universal material class with a ton of parameters; for this program the material needs to do two things:
  1. produce a scattered ray (or say it absorbed the incident ray, the ray of light that strikes a surface)
  2. if scattered, say how much the ray should be attenuated (reduction in intensity of a ray as it strikes a material, caused by absorption or scattering)
- the `material` class in `material.h` performs these operations

### 10.2 A Data Structure to Describe Ray-Object Intersections

- `hit_record` is used to store a bunch of information and avoid a bunch of arguments
  - we add a material pointer that will point at the sphere's material when the sphere was first initialized (consequently `sphere.h` gets some extra material fields)

### 10.3 Modeling Light Scatter and Reflectance

- *albedo* defines *fractional reflectance*; it varies with material color and can also vary with incident viewing direction (the direction of the incoming ray)
- Lambertian (diffuse) reflectance can either always scatter and attenuate light according to its reflectance *R* or it can sometimes scatter (with probability 1-*R*) with no attenuation (where a ray that isn't scattered is just absorbed into the material); it can also be a mixture of both
  - the authors choose to always scatter
  - the `lambertian` class is in `material.h`

## Miscellaneous

- C++ references are aliases; they do not have their own address; taking the address using `&` gives the address of the referent
- #include preprocessor directives' order will matter in this project! Just the way the authors have set it up, but be careful when modifying any #includes!
