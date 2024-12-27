
import os
import glob
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def extract_timestamps(log_file_path):
    timestamps = []
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                try:
                    timestamp_str = line.split(']')[0][1:]
                    timestamps.append(datetime.fromisoformat(timestamp_str))
                except ValueError as e:
                    print(f"Error parsing line: {line}")
    return timestamps

def analyze_timestamps(timestamps):
    if len(timestamps) < 2:
        return []

    differences = [(t2 - t1).total_seconds() for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    
    return differences

def main(log_folder):
    all_differences = []

    # Process each .log file in the given folder
    for log_file_path in glob.glob(os.path.join(log_folder, "*.log")):
        print(f"Processing file: {log_file_path}")
        timestamps = extract_timestamps(log_file_path)
        differences = analyze_timestamps(timestamps)
        all_differences.extend(differences)

    # Compute statistics
    if all_differences:
        stddev = np.std(all_differences)
        min_diff = np.min(all_differences)
        max_diff = np.max(all_differences)

        print(f"Standard Deviation: {stddev}")
        print(f"Minimum Difference: {min_diff}")
        print(f"Maximum Difference: {max_diff}")

        # Plot a histogram of the differences
        plt.hist(all_differences, bins=100, edgecolor='k', alpha=0.7)
        plt.title('Histogram of Time Differences Between Log Entries')
        plt.xlabel('Time Difference (seconds)')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("No differences found. Check if log files have enough entries.")

if __name__ == "__main__":
    log_folder = "./logs"  # Change this path to your log folder path
    main(log_folder)