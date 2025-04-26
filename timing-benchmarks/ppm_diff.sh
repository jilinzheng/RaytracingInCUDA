#!/bin/bash

# This script takes two input directories containing PPM files,
# pairs the files based on their modification date (assuming corresponding
# files have the same name and were created/modified in a similar order),
# executes a 'ppm_diff' program for each pair, and saves the output
# to a specified output directory with a prefixed filename.

# --- Configuration ---
# Set the path to your ppm_diff executable
PPM_DIFF_EXEC="../src/ppm_diff/ppm_diff"

# --- Input Validation ---
# Check if the correct number of command line arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_dir1> <input_dir2> <output_dir>"
    echo "  <input_dir1>: The first directory containing PPM files."
    echo "  <input_dir2>: The second directory containing PPM files."
    echo "  <output_dir>: The directory where output files will be saved."
    exit 1
fi

# Assign input arguments to variables
INPUT_DIR1="$1"
INPUT_DIR2="$2"
OUTPUT_DIR="$3"

# Validate that input directories exist
if [ ! -d "$INPUT_DIR1" ]; then
    echo "Error: Input directory 1 '$INPUT_DIR1' not found."
    exit 1
fi
if [ ! -d "$INPUT_DIR2" ]; then
    echo "Error: Input directory 2 '$INPUT_DIR2' not found."
    exit 1
fi

# Validate that the ppm_diff executable exists and is executable
if [ ! -x "$PPM_DIFF_EXEC" ]; then
    echo "Error: Executable '$PPM_DIFF_EXEC' not found or not executable."
    echo "Please ensure '$PPM_DIFF_EXEC' is in your PATH or provide the full path."
    exit 1
fi

# --- Setup Output Directory ---
# Create the output directory if it doesn't exist.
# The -p flag prevents an error if the directory already exists.
mkdir -p "$OUTPUT_DIR" || { echo "Error: Could not create output directory '$OUTPUT_DIR'."; exit 1; }

# --- Get and Sort Files ---
# Get lists of .ppm files from each directory, sorted by modification time (oldest first).
# We use `ls -tr` for sorting by time (reverse order, i.e., oldest first).
# `mapfile -t` reads the output lines into an array, handling spaces in filenames.
# We check if any .ppm files exist before trying to list them.

if ! ls "$INPUT_DIR1"/*.ppm >/dev/null 2>&1; then
    echo "Error: No .ppm files found in input directory 1 '$INPUT_DIR1'."
    exit 1
fi
if ! ls "$INPUT_DIR2"/*.ppm >/dev/null 2>&1; then
    echo "Error: No .ppm files found in input directory 2 '$INPUT_DIR2'."
    exit 1
fi

mapfile -t files1 < <(ls -tr "$INPUT_DIR1"/*.ppm)
mapfile -t files2 < <(ls -tr "$INPUT_DIR2"/*.ppm)

# --- Verify File Counts ---
# Check if the number of files in both directories is the same
num_files="${#files1[@]}"
if [ "$num_files" -eq 0 ]; then
    echo "No .ppm files found in either directory."
    exit 0 # Exit gracefully if no files are found
fi

if [ "$num_files" -ne "${#files2[@]}" ]; then
    echo "Error: Number of files in input directories do not match."
    echo "Files found in '$INPUT_DIR1': $num_files"
    echo "Files found in '$INPUT_DIR2': ${#files2[@]}"
    exit 1
fi

echo "Found $num_files pairs of files to process."

# --- Process File Pairs ---
# Loop through the files using an index
for ((i=0; i<num_files; i++)); do
    # Get the full path for the corresponding file in each directory
    file1="${files1[$i]}"
    file2="${files2[$i]}"

    # Extract just the filename from the full path
    filename1=$(basename "$file1")
    filename2=$(basename "$file2")

    # Optional: Verify that the filenames match.
    # This is a good check to ensure the sorting worked as expected
    # and corresponding files have identical names.
    if [ "$filename1" != "$filename2" ]; then
        echo "Warning: Filenames do not match for pair $i based on sorted order:"
        echo "  '$file1' vs '$file2'"
        echo "  Processing anyway, assuming sorted order is correct."
        # If you require strict filename matching, uncomment the next line:
        # exit 1
    fi

    # Construct the output filename with the required prefix
    output_filename="float_double_diff_${filename1}"

    # Construct the full output file path in the output directory
    output_filepath="${OUTPUT_DIR}/${output_filename}"

    echo "Processing pair: '$file1' and '$file2'"
    echo "Output will be saved to: '$output_filepath'"

    # Execute the ppm_diff program with the two input files and the output file path
    "$PPM_DIFF_EXEC" "$file1" "$file2" "$output_filepath"

    # Check the exit status of the ppm_diff executable
    if [ "$?" -ne 0 ]; then
        echo "Error: '$PPM_DIFF_EXEC' failed for pair '$filename1'."
        echo "Output file '$output_filepath' may be incomplete or missing."
        # Decide whether to stop on the first error or continue with the rest
        # exit 1 # Uncomment to stop on error
    fi

    echo "Finished processing '$filename1'."
    echo "" # Add a blank line for readability

done

echo "All file pairs processed."
