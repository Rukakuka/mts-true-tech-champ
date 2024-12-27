import os
import glob
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def extract_distance_changes(log_file_path):
    # Dictionary to keep track of changes in distance measurements and their timestamps
    distance_timestamps = defaultdict(list)

    with open(log_file_path, 'r', encoding='utf-8') as file:
        previous_measurements = {}
        previous_timestamps = {}
        previous_values = {}

        for line in file:
            if line.strip():
                try:
                    # Extract timestamp and JSON-like data from the line
                    timestamp_str, data_str = line.split(' ', 1)
                    timestamp = datetime.fromisoformat(timestamp_str[1:-1])
                    measurements = json.loads(data_str.replace("'", "\""))
                    for key, current_value in measurements.items():
                        if key.endswith('_distance'):
                            try:
                                _ = previous_values[key]
                            except KeyError:
                                previous_values[key] = current_value

                            try:
                                _ = previous_timestamps[key]
                            except KeyError:
                                previous_timestamps[key] = timestamp

                            try:
                                _ = previous_measurements[key]
                            except KeyError:
                                previous_measurements[key] = measurements
                            if previous_values[key] != current_value:
                                time_delta = (
                                    timestamp - previous_timestamps[key]).total_seconds()
                                if (time_delta > 2):
                                    print(f'Too big time delta in {log_file_path=}, {line=}')
                                distance_timestamps[key].append(time_delta)
                                previous_values[key] = current_value
                                previous_timestamps[key] = timestamp
                                previous_measurements[key] = measurements[key]

                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Error parsing line: {line}")

    return distance_timestamps

def analyze_time_deltas(all_distance_timestamps):
    results = {}

    for key, time_deltas in all_distance_timestamps.items():
        if time_deltas:
            time_deltas_array = np.array(time_deltas)
            stddev = np.std(time_deltas_array)
            mean = np.mean(time_deltas_array)
            min_val = np.min(time_deltas_array)
            max_val = np.max(time_deltas_array)

            results[key] = {
                'stddev': stddev,
                'mean': mean,
                'min': min_val,
                'max': max_val
            }

    return results

def plot_histograms(all_distance_timestamps):
    # Get keys and their count
    keys = list(all_distance_timestamps.keys())
    n_keys = len(keys)

    # Create a 2x3 subplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()  # To easily iterate over them

    for i, key in enumerate(keys):
        if i < 6:  # Restrict the plotting to the first 6 keys if there are more
            if all_distance_timestamps[key]:
                axs[i].hist(all_distance_timestamps[key], bins=50, edgecolor='k', alpha=0.7)
                axs[i].set_title(f"Time Delta Histogram for {key}")
                axs[i].set_xlabel('Time Delta (seconds)')
                axs[i].set_ylabel('Frequency')
                axs[i].grid(True)

    # Hide any unused subplots if there are fewer than 6 keys
    for i in range(n_keys, 6):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

def main(log_folder):
    all_distance_timestamps = defaultdict(list)

    for log_file_path in glob.glob(os.path.join(log_folder, "*.log")):
        print(f"Processing file: {log_file_path}")
        distance_timestamps = extract_distance_changes(log_file_path)

        for key, time_deltas in distance_timestamps.items():
            all_distance_timestamps[key].extend(time_deltas)

    results =analyze_time_deltas(all_distance_timestamps)
    for key, stats in results.items():
        print(f"Time delta statistics for {key}:")
        print(f"  Standard Deviation: {stats['stddev']:.2f} sec")
        print(f"  Mean: {stats['mean']:.2f} sec")
        print(f"  Minimum: {stats['min']:.2f} sec")
        print(f"  Maximum: {stats['max']:.2f} sec")
        print()

    # Plot histograms for each sensor
    plot_histograms(all_distance_timestamps)

if __name__ == "__main__":
    log_folder = "./logs"  # Change this path to your log folder path
    main(log_folder)