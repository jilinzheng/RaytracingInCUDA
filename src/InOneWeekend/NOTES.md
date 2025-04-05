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
- use image width to calculate height to keep aspect ratio consistent
- viewport: virtual rectangle in 3D world that contains the grid of image pixel locations
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

## Miscellaneous

- C++ references are aliases; they do not have their own address; taking the address using `&` gives the address of the referent
