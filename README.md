# **CUDA** Ray Tracing in One Weekend

## How do I use this codebase?

This codebase contains a little more than one implementation of raytracing in CUDA.

Specifically, in addition to the serial CPU version of raytracing obtained from
[Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io),
this repo contains implementations of raytracing in CUDA using global memory, constant memory,
and texture memory. There are separate versions using floats and doubles for the global and constant
memory implementations, but just one implementation using floats for the texture memory
implementation.

There are also a number of scripts that are used for benchmarking and performance analysis, so the
following subsections are provided to explain what is what in this repo.

Note that all of this code has been tested on a RTX 3070 (mobile/laptop) on an Ubuntu machine.
I've included a [`deviceQuery.txt`](./deviceQuery.txt) file that shows the output upon running the device query CUDA
tool, detailing my GPU specifications.

I hope everything below will be somewhat comprehensible and useful! I was able to achieve over 150x
speedups over the serial baseline, so I hope others may find similar results.

### Building

To build (and rebuild) any of the CUDA implementations, several `rebuild_*.sh` bash scripts are
in the root directory of this repository. They rely on relative paths, so be sure to be in the same
root directory when running the scripts. Of course, you can also run the `nvcc` compilation commands
manually, just have a look in the scripts if you need a reference! Be sure to change the `-gencode`
parameters to those that match your GPU. If you use the script, the binaries will be outputted to
the respective implementations' source directory - again, look at the script if you're unsure of
where to find the outputs! Note that [`rebuild_base.sh`](./rebuild_base.sh) just uses the build CMAKE build instructions
from the original Ray Tracing in One Weekend repo.

### Execution

The serial base CPU is ran the same way as the original Ray Tracing in One Weekend, e.g.,
`build\Release\inOneWeekend > image.ppm`.

As for my CUDA implementations, since I did a lot of performance benchmarking, the programs are set
up to write the generated .ppm image into a file named by the configuration parameters passed into
the program, and output a set of timing results/data to the console. This contrasts the serial
baseline where there is no timing set up within the program and the actual pixel values for the
result .ppm image is outputted to the console (redirection is used to save to a file like above).

I have set up benchmarking scripts by the name of `{memory}\_{datatype}\_benchmark.sh`, e.g.,
[`global_double_benchmark.sh`](./global_double_benchmark.sh). All of the scripts do the same thing on the different implementations.
They set up a list of configuration variables, including the scene (there are 3 hard-coded scenes),
image widths, image heights, number of samples for each ray, number of bounces for each ray,
number of threads to use in a row of 2D CUDA thread blocks, and number of runs to perform for each
combination of configurations (multiple runs reduce outlier effects).

Given the `global_double_benchmark.sh` file and the configuration of:

- scene = 1
- width = 320
- height = 192
- samples = 10
- bounces = 25
- threads = 4
- runs = 5

**ONE** .ppm file will be generated (since each run will overwrite the previous) in the
directory where the benchmarking script is ran, named
`global_float_scene1_320x192_10samples_25bounces_4threadsPerBlockRow.ppm`.

**ONE** .csv file will be generated that contains the configuration parameters above, the times
taken for the entire `render` kernel to finish, and the times taken for the end-to-end execution of
the entire program. Of course, each run is recorded separately. It is also best to run these
benchmarking scripts from where they're located, in the root directory of the repository, since they
rely on relative paths. It may also make more sense to just run one of these scripts for a grasp of
what I'm trying to communicate in regards to the script output. Make sure you know where the .csv
file will be by taking a look at the script (check the `OUTPUT_DIR` and `CSV_FILENAME` variables)!

Of course, you can also run the individual implementations manually. The usage will look like this:

```bash
# Assuming you are in src/GlobalDoubleCUDAInOneWeekend
./global-double-cuda-raytrace \
    --scene_id 1\
    --width 320 \
    --height 192 \
    --samples 10 \
    --bounces 25 \
    --threads 4
```

Similarly, upon completing execution, this will produce the .ppm file named after the configuration
variables and output timing to the console. The console timing output is just two comma-separated
values, first the `render` kernel-only timing, then the end-to-end program execution timing.

### Source Code

You'll be able to find all of the source code, as well as the output binaries in the [`src`](./src)
directory. Each varying memory and datatype implementation has its own subdirectory there.
[`src/InOneWeekend`](./src/InOneWeekend/) is the serial baseline. The other subdirectories are
self-explanatory, except ppm_diff, which you'll find an explanation of in the next subsection.

You may also find an explanation to my design decisions in the [Refactoring Motivation](#refactoring-motivation) section.

### `ppm_diff`

[`ppm_diff`](./src/ppm_diff/ppm_diff.cpp) is a convenience program to get the diff between two .ppm files. This can be used to
compare the differences with using floats vs. doubles in the CUDA kernel, as well as verifying the
inherent randomness of raytracing. It is expected that no two .ppm diffs will be the same, but be
relatively close in pixel values and produce a rather dark image. It is also expected that the two
input files are of the same size, i.e., same number of pixels.

`ppm_diff` takes three .ppm files as arguments, e.g., image1.ppm, image2.ppm, and output_diff.ppm.
It calculates the differences between the pixels in image1.ppm and image2.ppm and writes it to
output_diff.ppm. Minor differences between the images thus result in *darker* areas in the output
image, while larger differences result in *brighter* areas in the output image.

[`scaled_ppm_diff`](./src/ppm_diff/scaled_ppm_diff.cpp) is the same thing, just with values scaled
between 0-255.

```bash
# Don't forget to compile! Assuming you are in src/ppm_diff:
g++ -o ppm_diff ppm_diff.cpp
# Usage, assuming you have an image1.ppm and image2.ppm in the current directory:
./ppm_diff image1.ppm image2.ppm output_diff.ppm
```

### Additional/Miscellaneous Benchmarking Artifacts

Finally, there are some unaddressed artifacts, primarily related to profiling and benchmarking.

[`kernel-profiling`](./kernel-profiling/) contains a number of files related to my profiling with
the NVIDIA Nsight Compute CLI (`ncu`) and [`profile.sh`](./profile.sh). The shell script may be
useful if you're looking to do some profiling, but the `kernel-profiling` directory is likely
useless for you.

[`timing-benchmarks`](./timing-benchmarks/) contains the .csv files I had generated using the
aforementioned benchmarking scripts for the varying CUDA implementations. There is also a
[`ppm_diff.sh`](./timing-benchmarks/ppm_diff.sh) that takes two directories of .ppm files and runs
the `ppm_diff` convenience program on them to conveniently generate a bunch of diffs. Additionally,
there is a [`process.py`](./timing-benchmarks/process.py) that processes the raw .csv files
generated by the benchmarking scripts into .csv files containing average times for the `render`
kernel and end-to-end execution for each combination of configuration parameters. I had used this
to generate some pretty plots for my presentation on this project (yes, this was a final project
for my high performance programming course). You can find some of these plots [**here**](https://docs.google.com/presentation/d/1mU_EZI6tlWw8bthmDZeLxPUUETzLyll1cROyK8wAw3w/edit?usp=sharing)!

Last but not least, the [`CMakeLists.txt`](./CMakeLists.txt) is just the CMake file inherited from
the original Ray Tracing in One Weekend repo, used to build the serial baseline.

## Refactoring Motivation

A great reference and starting point was Roger Allen's
[Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/).
The code certainly still works, but some thing don't fully align with the most recent Ray Tracing
in One Weekend book. That was one reason to refactor, aside from just better understanding every
part of the code myself.

A more technical reason, however, was that Roger Allen had chosen to create the world using one
thread. I felt that a more intuitive design would be creating the world on the host, and
transferring it to the device. As of writing (April 17, 2025), simply adding `__device__` qualifiers
where appropriate to the base version and transferring host-created hittable objects (its children,
really) to the device would not work, as
["It is not allowed to pass as an argument to a global function an object of a class with virtual functions."](https://forums.developer.nvidia.com/t/can-cuda-properly-handle-pure-virtual-classes/37588).

As a result, I ended up refactoring to a more C-like style, where I use a function-based approach
and structs, ridding much of the polymorphism, inheritance, abstract classes, etc. concepts of C++.
Well, at least for the hittables ([`hittable.h`](./src/CUDAInOneWeekend/hittable.h)).

## References

- [Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io?tab=readme-ov-file)
- [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- [cxxopts](https://github.com/jarro2783/cxxopts)
