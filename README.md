# **CUDA** Ray Tracing in One Weekend

## Refactoring Motivation

A great reference and starting point was Roger Allen's [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/).
The code certainly still works, but some thing don't fully align with the most recent Ray Tracing
in One Weekend book. That was one reason to refactor, aside from just better understanding every part
of the code myself.

A more technical reason, however, was that Roger Allen had chosen to create the world using one thread.
I felt that a more intuitive design would be creating the world on the host, and transferring it to the
device. As of writing (April 17, 2025), simply adding `__device__` qualifiers where appropriate to the 
base version and transferring host-created hittable objects (its children, really) to the device would
not work, as ["It is not allowed to pass as an argument to a global function an object of a class with
virtual functions."](https://forums.developer.nvidia.com/t/can-cuda-properly-handle-pure-virtual-classes/37588).

As a result,I ended up refactoring to a more C-like style, where I use a function-based approach and 
structs, ridding much of the polymorphism, inheritance, abstract classes, etc. concepts of C++.
Well, at least for the hittables ([`hittable.h`](./src/CUDAInOneWeekend/hittable.h)).

## Relevant Sections from the Original README

### Optimized Builds
CMake supports Release and Debug configurations. These require slightly different invocations
across Windows (MSVC) and Linux/macOS (using GCC or Clang). The following instructions will place
optimized binaries under `build/Release` and debug binaries (unoptimized and containing debug
symbols) under `build/Debug`:

On Windows:

```shell
$ cmake -B build
$ cmake --build build --config Release  # Create release binaries in `build\Release`
$ cmake --build build --config Debug    # Create debug binaries in `build\Debug`
```

On Linux / macOS:

```shell
# Configure and build release binaries under `build/Release`
$ cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
$ cmake --build build/Release

# Configure and build debug binaries under `build/Debug`
$ cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
$ cmake --build build/Debug
```

We recommend building and running the `Release` version (especially before the final render) for
the fastest results, unless you need the extra debug information provided by the (default) debug
build.

### CMake GUI on Windows
You may choose to use the CMake GUI when building on windows.

1. Open CMake GUI on Windows
2. For "Where is the source code:", set to location of the copied directory. For example,
   `C:\Users\Peter\raytracing.github.io`.
3. Add the folder "build" within the location of the copied directory. For example,
   `C:\Users\Peter\raytracing.github.io\build`.
4. For "Where to build the binaries", set this to the newly-created "build" directory.
5. Click "Configure".
6. For "Specify the generator for this project", set this to your version of Visual Studio.
7. Click "Done".
8. Click "Configure" again.
9. Click "Generate".
10. In File Explorer, navigate to build directory and double click the newly-created `.sln` project.
11. Build in Visual Studio.

If the project is succesfully cloned and built, you can then use the native terminal of your
operating system to simply print the image to file.

### Running The Programs

You can run the programs by executing the binaries placed in the build directory:

    $ build\Debug\inOneWeekend > image.ppm

or, run the optimized version (if you compiled with the release configuration):

    $ build\Release\inOneWeekend > image.ppm

The generated PPM file can be viewed directly as a regular computer image, if your operating system
supports this image type. If your system doesn't handle PPM files, then you should be able to find
PPM file viewers online. We like [ImageMagick][].
