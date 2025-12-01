#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import csv
import itertools
from pathlib import Path
from datetime import datetime

# --- Configuration Constants ---
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
BUILD_DIR = PROJECT_ROOT / "build"
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
SCALED_PPM_DIFF_EXEC = SRC_DIR / "ppm_diff" / "scaled_ppm_diff"

# Compiler Flags (derived from your rebuild_*.sh files)
NVCC_FLAGS = [
    "-O3",
    "-gencode",
    "arch=compute_86,code=sm_86",
    # "-gencode", "arch=compute_70,code=sm_70",
    "-rdc=true",
    "--fmad=false",  # <--- ADD THIS to disable Fused Multiply-Add
]

# Target Definitions
# Maps a friendly name to specific source files and output paths
TARGETS = {
    "cpu": {
        "type": "cmake",
        "dir": SRC_DIR / "InOneWeekend",
        "exec_name": "inOneWeekend",
    },
    "gpu_global_float": {
        "type": "nvcc",
        "src": SRC_DIR / "GlobalFloatCUDAInOneWeekend" / "main.cu",
        "out_dir": SRC_DIR / "GlobalFloatCUDAInOneWeekend",
        "exec_name": "global-float-cuda-raytrace",
    },
    "gpu_global_double": {
        "type": "nvcc",
        "src": SRC_DIR / "GlobalDoubleCUDAInOneWeekend" / "main.cu",
        "out_dir": SRC_DIR / "GlobalDoubleCUDAInOneWeekend",
        "exec_name": "global-double-cuda-raytrace",
    },
    "gpu_const_float": {
        "type": "nvcc",
        "src": SRC_DIR / "ConstFloatCUDAInOneWeekend" / "main.cu",
        "out_dir": SRC_DIR / "ConstFloatCUDAInOneWeekend",
        "exec_name": "const-float-cuda-raytrace",
    },
    "gpu_const_double": {
        "type": "nvcc",
        "src": SRC_DIR / "ConstDoubleCUDAInOneWeekend" / "main.cu",
        "out_dir": SRC_DIR / "ConstDoubleCUDAInOneWeekend",
        "exec_name": "const-double-cuda-raytrace",
    },
    "gpu_tex_float": {
        "type": "nvcc",
        "src": SRC_DIR / "TexFloatCUDAInOneWeekend" / "main.cu",
        "out_dir": SRC_DIR / "TexFloatCUDAInOneWeekend",
        "exec_name": "tex-float-cuda-raytrace",
    },
}


def run_command(cmd, cwd=None):
    """Helper to run shell commands and handle errors."""
    try:
        print(f"Running: {' '.join(str(c) for c in cmd)}")
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


# --- Subcommand: Build ---
def build_target(args):
    target_keys = TARGETS.keys() if "all" in args.targets else args.targets

    for key in target_keys:
        config = TARGETS.get(key)
        if not config:
            print(f"Unknown target: {key}")
            continue

        print(f"\n--- Building {key} ---")
        if config["type"] == "cmake":
            # Replicates rebuild_base.sh
            build_type = "Debug" if args.debug else "Release"
            cmake_build_path = BUILD_DIR / build_type

            run_command(
                [
                    "cmake",
                    "-B",
                    str(cmake_build_path),
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                ],
                cwd=PROJECT_ROOT,
            )
            run_command(["cmake", "--build", str(cmake_build_path)], cwd=PROJECT_ROOT)

        elif config["type"] == "nvcc":
            # Replicates rebuild_*_cuda.sh
            output_exec = config["out_dir"] / config["exec_name"]
            cmd = ["nvcc", str(config["src"]), "-o", str(output_exec)] + NVCC_FLAGS

            # if key == "gpu_tex_float":
            #     cmd.append("--ptxas-options=-v")

            run_command(cmd)
            print(f"Successfully built: {output_exec}")


# --- Subcommand: Benchmark ---
def run_benchmark(args):
    config = TARGETS.get(args.target)
    if not config:
        print(f"Error: Unknown target '{args.target}'")
        return

    # Determine executable path
    if config["type"] == "cmake":
        # CPU build location differs based on CMake setup
        exec_path = BUILD_DIR / "Release" / config["exec_name"]
    else:
        exec_path = config["out_dir"] / config["exec_name"]

    if not exec_path.exists():
        print(f"Error: Executable not found at {exec_path}. Please build first.")
        return

    BENCHMARK_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    csv_filename = BENCHMARK_DIR / f"{timestamp}_{args.target}_timing.csv"

    # Configuration Space
    # You can modify these defaults or pass them via CLI if expanded
    scenes = [1]
    # resolutions = [(320, 192), (480, 288), (640, 384), (960, 576), (1280, 768)]
    resolutions = [(1280, 768)]
    # sample_counts = [10, 100]
    sample_counts = [100]
    bounces = [25]  # Fixed at 25 per your script
    # threads = [4, 8, 16]
    threads = [8]
    runs_per_config = 1

    print(f"Starting benchmark for {args.target}...")
    print(f"Output will be saved to {csv_filename}")

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene_id",
                "width",
                "height",
                "samples",
                "bounces",
                "threads",
                "run",
                "render_only_time_ms",
                "end_to_end_time_ms",
            ]
        )

        # Itertools product avoids deep nesting
        combinations = itertools.product(
            scenes, resolutions, sample_counts, bounces, threads
        )

        for scene, (w, h), samp, bounce, th in combinations:
            print(
                f"Config: Scene={scene}, Res={w}x{h}, Samp={samp}, Bounce={bounce}, Threads={th}"
            )

            for run in range(1, runs_per_config + 1):
                cmd = [
                    str(exec_path),
                    "--scene_id",
                    str(scene),
                    "--width",
                    str(w),
                    "--height",
                    str(h),
                    "--samples",
                    str(samp),
                    "--bounces",
                    str(bounce),
                    "--threads",
                    str(th),
                ]

                try:
                    # Capture stdout to get timing data
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    # Assuming output is space separated: "render_ms, total_ms"
                    timing_output = result.stdout.strip().split(", ")

                    if len(timing_output) >= 2:
                        row = [
                            scene,
                            w,
                            h,
                            samp,
                            bounce,
                            th,
                            run,
                            timing_output[0],
                            timing_output[1],
                        ]
                        writer.writerow(row)
                        print(
                            f"  Run {run}: {timing_output[0]}ms / {timing_output[1]}ms"
                        )
                    else:
                        print(f"  Run {run}: Failed to parse output: {result.stdout}")

                except subprocess.CalledProcessError as e:
                    print(f"  Run {run}: Crash/Error. {e.stderr}")


# --- Subcommand: Verify (Diff) ---
def run_verify(args):
    # Replaces ppm_diff.sh
    if not SCALED_PPM_DIFF_EXEC.exists():
        print("Error: scaled_ppm_diff executable not found. Compile utility first.")
        return

    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    files1 = sorted(list(dir1.glob("*.ppm")))
    files2 = sorted(list(dir2.glob("*.ppm")))

    if len(files1) != len(files2):
        print(
            f"Warning: File counts mismatch! {dir1}: {len(files1)}, {dir2}: {len(files2)}"
        )

    # Iterate through min length to prevent crash
    limit = min(len(files1), len(files2))

    for i in range(limit):
        f1 = files1[i]
        f2 = files2[i]

        # Basic name check (optional, but good for sanity)
        if f1.name != f2.name:
            print(f"Warning: Comparing distinct filenames: {f1.name} vs {f2.name}")

        diff_name = f"diff_{f1.name}"
        out_path = out_dir / diff_name

        print(f"Comparing {f1.name}...")
        try:
            subprocess.run(
                [str(SCALED_PPM_DIFF_EXEC), str(f1), str(f2), str(out_path)], check=True
            )
        except subprocess.CalledProcessError:
            print(f"Failed to compare pair {i}")


# --- Subcommand: Profile ---
def run_profile(args):
    # Replaces profile.sh
    config = TARGETS.get(args.target)
    if not config:
        print(f"Error: Unknown target '{args.target}'")
        return

    exec_path = config["out_dir"] / config["exec_name"]
    output_report = f"{args.target}-render-profile"

    # Construct Nsight Compute command
    cmd = [
        "ncu",
        "--set",
        "detailed",
        "-k",
        "render",
        "-o",
        output_report,
        "--force-overwrite",  # Safety for re-runs
        str(exec_path),
        "--scene_id",
        "1",  # Defaulting to scene 1 for profiling
    ]

    print(f"Profiling {args.target} with Nsight Compute...")
    run_command(cmd)


# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="Raytracing Automation Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build Parser
    p_build = subparsers.add_parser("build", help="Compile targets")
    p_build.add_argument(
        "targets", nargs="+", help="Target names (e.g., gpu_global_float) or 'all'"
    )
    p_build.add_argument(
        "--debug", action="store_true", help="Build debug version (CMake only)"
    )
    p_build.set_defaults(func=build_target)

    # Benchmark Parser
    p_bench = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    p_bench.add_argument("target", help="Target to benchmark (e.g., gpu_global_float)")
    p_bench.set_defaults(func=run_benchmark)

    # Verify Parser
    p_verify = subparsers.add_parser("verify", help="Compare PPM outputs")
    p_verify.add_argument("dir1", help="First directory of PPMs")
    p_verify.add_argument("dir2", help="Second directory of PPMs")
    p_verify.add_argument("out_dir", help="Directory to save diff images")
    p_verify.set_defaults(func=run_verify)

    # Profile Parser
    p_profile = subparsers.add_parser("profile", help="Profile with Nsight Compute")
    p_profile.add_argument("target", help="Target to profile")
    p_profile.set_defaults(func=run_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
