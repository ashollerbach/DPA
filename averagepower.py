import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Specify folder path containing CSV files
folder_path = '0a4d6f8107274375cf0a549995fe79f0'  # Replace with your folder path

# Initialize lists to store data
all_ch1 = []
time = None

# Read all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
for file in csv_files:
    # Read CSV, skip metadata rows
    data = pd.read_csv(file, skiprows=12)
    data = data.iloc[1:]  # Remove first row

    # Extract TIME and CH1 columns
    if time is None:
        time = pd.to_numeric(data['Labels'], errors='coerce')
    ch1 = pd.to_numeric(data['Unnamed: 1'], errors='coerce')

    # Store CH1 data
    all_ch1.append(ch1)

# Convert list to DataFrame and compute average
ch1_df = pd.DataFrame(all_ch1)
average_ch1 = ch1_df.mean(axis=0)


# Standardize average CH1 to mean=0, std=1
mu = np.mean(average_ch1)
sigma = np.std(average_ch1)
ch1_standardized = (average_ch1 - mu) / sigma

#set limit to improve visibility
limit = 1000
range = 1000
time = time[limit:limit + range]
ch1_standardized = ch1_standardized[limit :limit + range]
# Plot TIME vs. standardized average CH1
plt.plot(time, ch1_standardized, 'b-', label='Standardized Average CH1')
plt.xlabel('Time (s)')
plt.ylabel('Standardized Current (z-score)')
plt.title('Standardized Average CH1 vs. Time')
plt.grid(True)
plt.legend()

# Save plot
plt.savefig('average_power_trace.png')