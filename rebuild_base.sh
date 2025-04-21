#!/usr/bin/bash

cmake -B build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug

time ./build/Release/inOneWeekend > base_image.ppm

