#!/usr/bin/bash

# --- Configuration Variables ---
# SCENE_IDS=(1 2 3)
SCENE_IDS=(1)
WIDTHS=(320 480 640 960 1280)
HEIGHTS=(192 288 384 576 768)
SAMPLES=(10 100)
BOUNCES=(25)
THREADS=(4 8 16 32)
RUNS=5 # Number of times to run each combination

# Output directory and filename for the single CSV file
OUTPUT_DIR="./benchmarks"
CSV_FILENAME="${OUTPUT_DIR}/250427_gpu_const_double_timing.csv"

# --- Setup ---
# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# --- Write CSV Header (only once) ---
# Use '>' to create or overwrite the file and write the header row.
# Adjust the "othermetric1,othermetric2,..." part to match the actual
# column names for the data your executable outputs.
echo "scene_id,width,height,samples,bounces,threads,run,render_only_time_ms,end_to_end_time_ms" > "$CSV_FILENAME"

# --- Nested Loops for Combinations and Runs ---

# Loop through all threads (per block row)
for threads in ${THREADS[@]}; do
  # Loop through all scene IDs
  for scene_id in "${SCENE_IDS[@]}"; do
    # Loop through all sample counts
    for samples in "${SAMPLES[@]}"; do
      # Loop through all bounce counts
      for bounces in "${BOUNCES[@]}"; do
        # --- Loop through paired width and height using indices ---
        # We loop through the indices of the WIDTHS array (assuming HEIGHTS has same size)
        for i in "${!WIDTHS[@]}"; do
          # Get the corresponding width and height using the current index
          width="${WIDTHS[$i]}"
          height="${HEIGHTS[$i]}"
          echo "--- Starting runs for: Scene=$scene_id, Res=${width}x${height}, Samples=$samples, Bounces=$bounces, Threads=$threads ---"
          # --- Inner Loop for Multiple Runs ---
          # Loop RUNS times for the current combination
          for run_num in $(seq 1 $RUNS); do
            echo "  Running $run_num of $RUNS..."

            # Execute the raytracing program and CAPTURE its standard output.
            # Replace './your_raytrace_executable' with your actual command.
            # Ensure your executable prints the data you want to capture to standard output.
            # Command substitution $(...) captures the output.
            EXECUTABLE_OUTPUT=$(
            ./src/ConstDoubleCUDAInOneWeekend/const-double-cuda-raytrace \
                --scene_id "$scene_id" \
                --width "$width" \
                --height "$height" \
                --samples "$samples" \
                --bounces "$bounces" \
                --threads "$threads"
            )

            # --- Process Executable Output and Append to CSV ---
            # We need to format the captured output to fit into the CSV columns.
            # Assuming your executable outputs space-separated values (e.g., "5.123 0.987")
            # If your executable outputs comma-separated values or a different format,
            # you will need to adjust the processing below.

            # Example: Replacing spaces in output with commas for CSV
            # PROCESSED_OUTPUT=$(echo "$EXECUTABLE_OUTPUT" | tr ' ' ',')

            # Construct the full CSV line with all parameters and the processed output,
            # then append it to the single CSV file.
            echo "${scene_id},${width},${height},${samples},${bounces},${threads},${run_num},${EXECUTABLE_OUTPUT}" >> "$CSV_FILENAME"
          done # End of RUNS loop
        echo "--- Finished runs for: Scene=$scene_id, Res=${width}x${height}, Samples=$samples, Bounces=$bounces, Threads=$threads ---"
        echo "" # Add a blank line for readability between combinations in the log
        done # End of WidthxHeight loop
      done # End of BOUNCES loop
    done # End of SAMPLES loop
  done # End of SCENE_IDS loop
done # End of THREADS loop

echo "All combinations and runs complete. All results saved in '$CSV_FILENAME'."

echo "Finished processing all combinations."
