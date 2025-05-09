# dpa_key_recovery.py
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

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

def plot_average_traces_from_folders(trails_path, output_filename='combined_power_analysis_plots.png', limit=1600, trace_range=3500):
    """
    Processes power traces from subfolders, creating two side-by-side subplots:
    1. Overlayed standardized average power traces.
    2. Vertically stacked dot plot of standardized maximum values from each trace's 
       specified range, colored by trace, with jitter for visibility.

    Args:
        trails_path (str): Path to the main directory containing subfolders of traces.
        output_filename (str): Name of the file to save the combined plot.
        limit (int): Number of initial samples to skip in each trace.
        trace_range (int): Number of samples to include in the plot after the limit.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7)) 
    
    master_time_vector = None 
    # Stores (folder_name, max_value_of_standardized_trace_segment, color_of_trace_on_ax1)
    max_values_data = [] 

    subfolders = [f.path for f in os.scandir(trails_path) if f.is_dir()]

    if not subfolders:
        print(f"No subfolders found in {trails_path}. Cannot generate plots.")
        plt.close(fig) 
        return

    print(f"Found subfolders: {[os.path.basename(sf) for sf in subfolders]}")

    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        print(f"\nProcessing folder: {folder_name}")

        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        if not csv_files:
            print(f"  No CSV files found in {folder_path}. Skipping this folder.")
            continue
        
        print(f"  Found {len(csv_files)} CSV files.")
        all_ch1_for_folder = []
        current_folder_time_vector = None

        for i, file_path in enumerate(csv_files):
            try:
                data = pd.read_csv(file_path, skiprows=12)
                if data.empty: continue
                data = data.iloc[1:] 
                if data.empty: continue
                if 'Labels' not in data.columns or 'Unnamed: 1' not in data.columns: continue
                
                time_col = pd.to_numeric(data['Labels'], errors='coerce')
                ch1_col = pd.to_numeric(data['Unnamed: 1'], errors='coerce')
                valid_indices = time_col.notna() & ch1_col.notna()
                time_col = time_col[valid_indices]
                ch1_col = ch1_col[valid_indices]
                if ch1_col.empty: continue

                if current_folder_time_vector is None:
                    current_folder_time_vector = time_col
                
                if len(ch1_col) != len(current_folder_time_vector):
                    min_len = min(len(ch1_col), len(current_folder_time_vector))
                    ch1_col = ch1_col.iloc[:min_len]
                    if i == 0: 
                        current_folder_time_vector = current_folder_time_vector.iloc[:min_len]
                all_ch1_for_folder.append(ch1_col)
            except Exception as e:
                print(f"  Error processing file {os.path.basename(file_path)}: {e}")
                continue
        
        if not all_ch1_for_folder: continue
        try:
            min_len_this_folder = min(len(trace) for trace in all_ch1_for_folder)
        except ValueError: continue
        all_ch1_aligned = [trace.iloc[:min_len_this_folder].values for trace in all_ch1_for_folder]
        
        if current_folder_time_vector is not None:
             current_folder_time_vector = current_folder_time_vector.iloc[:min_len_this_folder]
        else: 
            print(f"  Could not establish a time vector for folder {folder_name}. Skipping.")
            continue
        if not all_ch1_aligned: 
            print(f"  No traces could be aligned for folder {folder_name}. Skipping.")
            continue

        average_ch1 = np.mean(np.vstack(all_ch1_aligned), axis=0)
        if average_ch1.size == 0: 
            print(f"  Average CH1 is empty for folder {folder_name}. Skipping.")
            continue

        mu = np.mean(average_ch1)
        sigma = np.std(average_ch1)
        ch1_standardized = (average_ch1 - mu) / sigma if sigma != 0 else average_ch1 - mu
        
        if len(current_folder_time_vector) < limit + trace_range or len(ch1_standardized) < limit + trace_range:
            print(f"  Folder {folder_name} has insufficient data points ({len(ch1_standardized)}) for the specified limit ({limit}) and range ({trace_range}). Skipping.")
            continue

        time_to_plot = current_folder_time_vector.iloc[limit : limit + trace_range]
        trace_to_plot = ch1_standardized[limit : limit + trace_range]
        if time_to_plot.empty or trace_to_plot.size == 0: 
            print(f"  Trace for {folder_name} is empty after slicing. Skipping.")
            continue
            
        folder_color_for_ax1 = None 
        if master_time_vector is None: 
            master_time_vector = time_to_plot.values 
        
        if len(time_to_plot.values) == len(master_time_vector) and np.allclose(time_to_plot.values, master_time_vector):
            line, = ax1.plot(master_time_vector, trace_to_plot, label=f'{folder_name}') 
            folder_color_for_ax1 = line.get_color() 
            print(f"  Plotted trace for {folder_name} with color {folder_color_for_ax1}.")
            if trace_to_plot.size > 0: 
                max_values_data.append((folder_name, np.max(trace_to_plot), folder_color_for_ax1))
        else:
            print(f"  Warning: Time axis for {folder_name} mismatch. Skipping for trace overlay plot.")
            # Only add to max_values_data if it was plotted on ax1 to ensure color consistency for ax2.

    # --- Configure and finalize the first subplot (Overlayed Traces) ---
    if not ax1.lines: 
        ax1.text(0.5, 0.5, 'No trace data plotted.', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        print("\nNo traces were plotted on the first subplot.")
    else:
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Standardized Current (z-score)')
        ax1.set_title(f'Overlayed Standardized Average Power Traces\n(Samples {limit}-{limit+trace_range-1})')
        ax1.grid(True)
        ax1.legend(loc='best', fontsize='small')

    # --- Create and configure the second subplot (Vertically Stacked Colored Dots for Max Values) ---
    if not max_values_data:
        ax2.text(0.5, 0.5, 'No max value data to plot.', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        print("\nNo data available to plot standardized maximum values as dots.")
    else:
        folder_names_for_dot_plot = [item[0] for item in max_values_data]
        raw_max_vals = np.array([item[1] for item in max_values_data])
        colors_for_dot_plot = [item[2] for item in max_values_data]

        mu_max_vals = np.mean(raw_max_vals)
        sigma_max_vals = np.std(raw_max_vals)
        
        if sigma_max_vals == 0:
            print("\nWarning: Standard deviation of collected max values is zero. Plotting centered max values.")
            standardized_max_vals_to_plot = raw_max_vals - mu_max_vals
            y_label_max_plot = 'Centered Max Power Value in Range'
        else:
            standardized_max_vals_to_plot = (raw_max_vals - mu_max_vals) / sigma_max_vals
            y_label_max_plot = 'Standardized Max Power Value in Range (z-score)'

        # Create the dot plot on ax2
        # Jitter for x-coordinates to prevent perfect overlap
        jitter_strength = 0.05 # Adjust for more/less spread
        x_coords_for_dots = np.random.normal(loc=0, scale=jitter_strength, size=len(folder_names_for_dot_plot))

        for i, folder_name in enumerate(folder_names_for_dot_plot):
            ax2.scatter(x_coords_for_dots[i], # X-coordinate with jitter around 0
                        standardized_max_vals_to_plot[i], # Y-coordinate is the standardized max value
                        color=colors_for_dot_plot[i], 
                        s=100, # Size of the dots
                        label=folder_name, 
                        alpha=0.7, 
                        edgecolors='k') # Edge color for dots
        
        ax2.set_xticks([]) # Remove x-axis ticks as they are not meaningful here
        ax2.set_xlabel('Color-Coded Traces') # More generic x-axis label
        ax2.set_ylabel(y_label_max_plot)
        ax2.set_title(f'Standardized Max Power Values per Trace\n(Samples {limit}-{limit+trace_range-1})')
        ax2.grid(axis='y', linestyle='--')
        ax2.legend(title='Folder/Trace', loc='best', fontsize='small', markerscale=0.7)

    # --- Finalize and save/show the combined plot ---
    fig.suptitle('Power Trace Analysis', fontsize=16, y=1.02) 
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    
    try:
        plt.savefig(output_filename)
        print(f"\nCombined plot saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving combined plot: {e}")
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    trails_main_path = 'trails' 
    plot_limit = 1600 
    plot_trace_range = 125 
    combined_plot_file = 'combined_power_analysis_plots.png'

    dummy_data_created_this_run = False
    if not os.path.exists(trails_main_path):
        print(f"'{trails_main_path}' not found. Creating dummy data for testing...")
        os.makedirs(trails_main_path, exist_ok=True)
        num_dummy_folders = 4 
        traces_per_dummy_folder = 3
        num_dummy_samples = 6000 
        base_time_dummy = np.linspace(0, num_dummy_samples / 100000.0, num_dummy_samples) 

        for i in range(num_dummy_folders):
            dummy_folder_name = f"dummy_plaintext_{i:02x}" 
            subfolder_path = os.path.join(trails_main_path, dummy_folder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            for j in range(traces_per_dummy_folder):
                csv_path = os.path.join(subfolder_path, f"trace_{j}.csv")
                with open(csv_path, 'w') as f:
                    for k in range(12): f.write(f"Metadata line {k+1}\n")
                    f.write("Header,Labels,Unnamed: 1,OtherColumn\n") 
                    f.write("This,is,a,dummyrow\n") 
                    
                    noise = np.random.randn(num_dummy_samples) * (0.5 + i*0.1) 
                    signal_amplitude = 5 + i*2 
                    signal = signal_amplitude * np.sin(2 * np.pi * (i + 1) * base_time_dummy / (num_dummy_samples/100)) + (j * 0.2) 
                    trace_data = signal + noise
                    for k_sample in range(num_dummy_samples):
                        f.write(f"Sample{k_sample},{base_time_dummy[k_sample]:.6f},{trace_data[k_sample]:.6f},0\n")
        print(f"Dummy data created in '{trails_main_path}'.")
        dummy_data_created_this_run = True

    if dummy_data_created_this_run:
        print("Using adjusted plot parameters for dummy data.")
        plot_limit = 100
        if num_dummy_samples - plot_limit < plot_trace_range :
             plot_trace_range = max(50, num_dummy_samples - plot_limit - 1) 
        print(f"Adjusted dummy data: plot_limit={plot_limit}, plot_trace_range={plot_trace_range}")

    print(f"\nStarting trace plotting from: {os.path.abspath(trails_main_path)}")
    print(f"Plotting samples from {plot_limit} to {plot_limit + plot_trace_range - 1}")
    plot_average_traces_from_folders(
        trails_main_path, 
        output_filename=combined_plot_file,
        limit=plot_limit, 
        trace_range=plot_trace_range
    )
    print("\nScript finished.")
