import numpy as np
import pandas as pd
import glob
import os

# AES forward S-box lookup table
sbox = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

def hamming_weight(n):
    """Calculate Hamming weight of a byte."""
    return bin(n).count('1')

def load_and_normalize_traces(trails_path, plaintexts, limit=1600, trace_range=3500):
    """Load, average, and normalize power traces from subfolders."""
    traces_list = []
    time = None

    # Iterate through plaintexts to match subfolders
    for plaintext in plaintexts:
        # Convert plaintext bytes to hex string for folder name
        folder_name = plaintext.hex()
        folder_path = os.path.join(trails_path, folder_name)

        # Read all CSV files in the subfolder
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {folder_path}")

        # Load and average traces for this plaintext
        ch1_list = []
        for file in csv_files:
            data = pd.read_csv(file, skiprows=12)
            data = data.iloc[1:]  # Remove first row
            if time is None:
                time = pd.to_numeric(data['Labels'], errors='coerce').dropna()
            ch1 = pd.to_numeric(data['Unnamed: 1'], errors='coerce').dropna()
            ch1_list.append(ch1)

        # Compute average waveform
        ch1_df = pd.DataFrame(ch1_list)
        average_ch1 = ch1_df.mean(axis=0)

        # Standardize average waveform
        mu = np.mean(average_ch1)
        sigma = np.std(average_ch1)
        if sigma == 0:
            raise ValueError(f"Zero standard deviation for plaintext {folder_name}")
        ch1_normalized = (average_ch1 - mu) / sigma

        # Store normalized average
        traces_list.append(ch1_normalized)

    # Convert to NumPy array and slice
    traces = np.array(traces_list)
    time = time[limit:limit + trace_range]
    traces = traces[:, limit:limit + trace_range]

    # Verify data
    if traces.shape[0] == 0 or traces.shape[1] == 0:
        raise ValueError("Empty traces after slicing")
    if len(time) != traces.shape[1]:
        raise ValueError("Time and traces length mismatch")
    if traces.shape[0] != len(plaintexts):
        raise ValueError("Number of traces does not match number of plaintexts")

    return time, traces

def dpa_attack(plaintexts, traces):
    """Perform DPA attack to recover AES-128 key."""
    num_traces, trace_length = traces.shape
    num_bytes = 16  # AES-128 has 16 key bytes
    recovered_key = []

    for byte_idx in range(num_bytes):
        # Hypothetical power consumption for each key guess
        hypo_power = np.zeros((256, num_traces), dtype=np.uint8)
        for key_guess in range(256):
            for t in range(num_traces):
                # Compute S-box output for plaintext XOR key_guess
                intermediate = sbox[plaintexts[t, byte_idx] ^ key_guess]
                hypo_power[key_guess, t] = hamming_weight(intermediate)

        # Compute correlation between hypothetical and measured power
        max_corr = 0
        best_key = 0
        for key_guess in range(256):
            for sample in range(trace_length):
                corr = np.corrcoef(hypo_power[key_guess], traces[:, sample])[0, 1]
                if not np.isnan(corr) and abs(corr) > max_corr:
                    max_corr = abs(corr)
                    best_key = key_guess

        recovered_key.append(best_key)
        print(f"Byte {byte_idx}: Key guess {hex(best_key)} (Correlation: {max_corr:.4f})")

    return bytes(recovered_key)

# Example usage
trails_path = 'trails'  # Replace with your path
plaintext_file = 'plaintexts.txt'  # Replace with your plaintext file

# Load plaintexts (32-char hex strings, one per line)
plaintexts = np.array([bytes.fromhex(line.strip()) for line in open(plaintext_file)])
# Load and normalize averaged traces
time, traces = load_and_normalize_traces(trails_path, plaintexts, limit=1600, trace_range=3500)

# Run DPA attack
key = dpa_attack(plaintexts, traces)
print(f"Recovered key: {[hex(b) for b in key]}")