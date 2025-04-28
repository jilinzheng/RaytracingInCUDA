import pandas as pd

# Define the input and output file names
input_filenames= [
    # '250427_gpu_global_double_timing.csv',
    # '250427_gpu_const_double_timing.csv',
    # '250427_gpu_global_float_timing.csv',
    # '250427_gpu_const_float_timing.csv'
    '250427_gpu_tex_float_timing.csv'
]
output_filenames = [
    # '250427_avg_gpu_global_double_timing.csv',
    # '250427_avg_gpu_const_double_timing.csv',
    # '250427_avg_gpu_global_float_timing.csv',
    # '250427_avg_gpu_const_float_timing.csv'
    '250427_avg_gpu_tex_float_timing.csv'
]

# Define the columns to group by
group_cols = ['scene_id', 'width', 'height', 'samples', 'bounces', 'threads']

for i in range(len(input_filenames)):
    try:
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(input_filenames[i])

        # Group by the specified columns and calculate the mean of the time columns
        averaged_data = df.groupby(group_cols)[['render_only_time_ms', 'end_to_end_time_ms']].mean().reset_index()

        # Rename the average columns
        averaged_data = averaged_data.rename(columns={
            'render_only_time_ms': 'avg_render_only_time_ms',
            'end_to_end_time_ms': 'avg_end_to_end_time_ms'
        })

        # Save the new DataFrame to a new CSV file
        averaged_data.to_csv(output_filenames[i], index=False)

        print(f"Averaged data saved to '{output_filenames[i]}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filenames[i]}' not found. Make sure the CSV file is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")
