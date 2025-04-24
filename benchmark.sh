#!/usr/bin/bash

mkdir -p ./benchmarks
OUTPUT_DIR=./benchmarks

FLOAT_EXE=./src/FloatCUDAInOneWeekend/float-cuda-raytrace
DOUBLE_EXE=./src/DoubleCUDAInOneWeekend/double-cuda-raytrace

SCENE_IDS=(1 2 3)
WIDTHS=(320 640 1280 2560)
HEIGHTS=(192 384 768 1536)
SAMPLES=(10 100)
BOUNCES=(25 50)
THREADS=8
RUNS=5

# Start the nested loops
for scene_id in "${SCENE_IDS[@]}"; do
    for width in "${WIDTHS[@]}"; do
        for height in "${HEIGHTS[@]}"; do
            for samples in "${SAMPLES[@]}"; do
                for bounces in "${BOUNCES[@]}"; do
                    # --- Per-Combination Setup ---
                    # Construct the unique filename for this combination of parameters
                    CSV_FILENAME="${OUTPUT_DIR}/scene${scene_id}_${width}x${height}_${samples}samples_${bounces}bounces.csv"

                    # Optional: Add a header row to the CSV file if it doesn't exist
                    # This is useful for making the CSV readable. Adjust the header based on your executable's output.
                    if [ ! -f "$CSV_FILENAME" ]; then
                        # Replace "Run_Number,Executable_Output..." with appropriate headers for your data
                        echo "Run,Render-Only Time(ms),End-to-End Time(ms)" >> "$CSV_FILENAME"
                    fi

                    echo "--- Starting runs for: Scene=$scene_id, Res=${width}x${height}, Samples=$samples, Bounces=$bounces ---"

                    # --- Inner Loop for Multiple Runs ---
                    # Loop RUNS times for the current combination
                    for run_num in $(seq 1 $RUNS); do
                        echo "  Running $run_num of $RUNS..."

                        # Execute your executable and append its output to the CSV file.
                        # Make sure your executable prints the data you want to capture to its standard output.
                        # We first append the run number and a comma, then append the executable's output.
                        # Ensure the executable's output starts with the data you want immediately after the comma.

                        echo -n "$run_num," >> "$CSV_FILENAME" # Append run number and a comma ( -n prevents newline)

                        # Execute the raytracing program and append its standard output to the CSV.
                        # Make sure the executable's output format matches what you expect after the comma.
                        # If the executable outputs multiple lines or complex formats, you might need to process its output first.
                        FLOAT_EXE \
                            --scene_id "$scene_id" \
                            --width "$width" \
                            --height "$height" \
                            --samples "$samples" \
                            --bounces "$bounces" >> "$CSV_FILENAME"

                        # Add a newline character after the executable's output if the executable itself doesn't add one
                        # or if you need a consistent newline after each run's data.
                        # echo "" >> "$CSV_FILENAME" # Uncomment this line if needed
                    done # End of RUNS loop

                    echo "--- Finished runs for: Scene=$scene_id, Res=${width}x${height}, Samples=$samples, Bounces=$bounces ---"
                    echo "" # Add a blank line for readability between combinations
                done
            done
        done
    done
done

echo "Finished processing all combinations."
