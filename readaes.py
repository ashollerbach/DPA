import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV data
data = pd.read_csv('aes_ch1_20250506132504400.csv', skiprows=11)  # Skip metadata rows

# Extract TIME and CH1 columns
time = data['TIME']
ch1 = data['CH1']

# Standardize CH1 to mean=0, std=1
mu = np.mean(ch1)
sigma = np.std(ch1)
ch1_standardized = (ch1 - mu) / sigma

# Plot TIME vs. standardized CH1
plt.plot(time, ch1_standardized, 'b-', label='Standardized CH1')
plt.xlabel('Time (s)')
plt.ylabel('Standardized Current (z-score)')
plt.title('Standardized CH1 vs. Time')
plt.grid(True)
plt.legend()

# Show plot
plt.show()