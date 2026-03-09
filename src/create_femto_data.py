import os
import pandas as pd
import numpy as np
import scipy.stats
import numpy.fft as fft
from tqdm import tqdm

print("--- Step 1: Creating FEMTO Sensor Fusion Dataset (Robust Version) ---")

# --- Configuration ---
# POINT THIS TO YOUR RAW FEMTO DATA FOLDER
# Based on your screenshots, your folder is named 'Training_set'
# and is located in the main project directory.
# Since this script is inside 'scripts001', we go UP one level (..) to find it.
RAW_DATA_DIR = 'C:\project file 002\Training_set' 

# Output directory for the processed .npy files
PROCESSED_DATA_DIR = 'C:\project file 002\processed data001' 

# Output Files
X_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'X_data_femto_fusion.npy')
Y_REG_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_reg_femto_fusion.npy')
Y_CLASS_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_class_femto_fusion.npy')

# Parameters
WINDOW_SIZE = 2560
SAMPLING_RATE = 25600
FAILURE_THRESHOLD = 50 # Last 50 files = "Failing"

# --- CRITICAL: Load Definitions ---
# Based on FEMTO-ST dataset documentation:
# Bearing1_1: Load 4000N
# Bearing1_2: Load 4000N
# Bearing2_1: Load 4200N
# Bearing2_2: Load 4200N
# Bearing3_1: Load 5000N
# Bearing3_2: Load 5000N
BEARING_LOADS = {
    'Bearing1_1': 4000,
    'Bearing1_2': 4000,
    'Bearing2_1': 4200,
    'Bearing2_2': 4200,
    'Bearing3_1': 5000,
    'Bearing3_2': 5000
}

def extract_features(segment, sampling_rate):
    # Time-Domain
    mean = np.mean(segment)
    std = np.std(segment)
    rms = np.sqrt(np.mean(segment**2))
    kurt = scipy.stats.kurtosis(segment)
    skew = scipy.stats.skew(segment)
    peak = np.max(np.abs(segment))
    crest = peak / rms if rms != 0 else 0
    time_features = [mean, std, rms, kurt, skew, peak, crest]
    
    # Frequency-Domain (FFT)
    freqs = np.fft.fftfreq(len(segment), d=1.0/sampling_rate)[:len(segment)//2]
    fft_mag = np.abs(np.fft.fft(segment))[:len(segment)//2]
    if len(freqs) > 0:
        fft_peak = freqs[np.argmax(fft_mag)]
    else:
        fft_peak = 0
    fft_mean = np.mean(fft_mag)
    fft_std = np.std(fft_mag)
    freq_features = [fft_peak, fft_mean, fft_std]
    
    return time_features + freq_features

def process_data():
    all_features = []
    all_rul = []
    all_status = []
    
    # Check if raw data dir exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"ERROR: Raw data directory not found at: {RAW_DATA_DIR}")
        return

    # Get list of bearings to process based on folder existence and our load definitions
    bearings = [b for b in os.listdir(RAW_DATA_DIR) if b.startswith('Bearing') and b in BEARING_LOADS]
    
    print(f"Found {len(bearings)} bearings to process in {RAW_DATA_DIR}")
    
    for bearing in tqdm(bearings, desc="Processing Bearings"):
        path = os.path.join(RAW_DATA_DIR, bearing)
        load_val = BEARING_LOADS[bearing] # Get the correct load
        
        # Get file lists
        # Sorting is crucial to keep time order correct
        acc_files = sorted([f for f in os.listdir(path) if f.startswith('acc') and f.endswith('.csv')])
        temp_files = sorted([f for f in os.listdir(path) if f.startswith('temp') and f.endswith('.csv')])
        
        count_acc = len(acc_files)
        count_temp = len(temp_files)

        # --- ROBUST MISMATCH HANDLING ---
        if count_acc == 0:
             print(f"Skipping {bearing}: No acceleration files found.")
             continue
        
        # Use the minimum count to ensure we always have pairs
        limit = min(count_acc, count_temp)
        
        if count_acc != count_temp:
            # This print confirms the script is handling the mismatch for you!
            print(f"\n[Info] Handling mismatch in {bearing}. Acc: {count_acc}, Temp: {count_temp}. Using first {limit} files.")
            
        # Process each file pair as one sample
        for i in range(limit):
            # 1. Load Data
            try:
                # Handle potential separator differences in CSVs (FEMTO can use ; or ,)
                acc_path = os.path.join(path, acc_files[i])
                acc_df = pd.read_csv(acc_path, header=None, sep=';')
                if acc_df.shape[1] < 6: 
                     acc_df = pd.read_csv(acc_path, header=None, sep=',')

                temp_path = os.path.join(path, temp_files[i])
                temp_df = pd.read_csv(temp_path, header=None, sep=';')
                if temp_df.shape[1] < 5:
                     temp_df = pd.read_csv(temp_path, header=None, sep=',')
            except Exception as e:
                # If a single file is corrupt, just skip it and keep going
                continue

            # 2. Get Segments
            # FEMTO: Col 4 is Horiz Vib, Col 5 is Vert Vib
            # Temp: Col 4 is Temperature
            try:
                vib_horiz = acc_df.iloc[:, 4].values
                vib_vert = acc_df.iloc[:, 5].values
                temp_val = temp_df.iloc[:, 4].mean() # Average temp for this file
            except IndexError:
                continue # Skip if columns are missing

            # 3. Extract Features
            feats_h = extract_features(vib_horiz, SAMPLING_RATE) # 10 features
            feats_v = extract_features(vib_vert, SAMPLING_RATE)  # 10 features
            
            # 4. Combine All Features (Sensor Fusion!)
            # [10 Horiz Vib] + [10 V-Vib] + [1 Temp] + [1 Load] = 22 Features
            sample_features = feats_h + feats_v + [temp_val] + [load_val]
            
            all_features.append(sample_features)
            
            # 5. Labels
            rul = limit - i # RUL in "number of files remaining"
            status = 1 if rul < FAILURE_THRESHOLD else 0 # 1 = Danger
            
            all_rul.append(rul)
            all_status.append(status)

    # Save results to disk
    if len(all_features) > 0:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        np.save(X_OUTPUT_FILE, np.array(all_features))
        np.save(Y_REG_OUTPUT_FILE, np.array(all_rul))
        np.save(Y_CLASS_OUTPUT_FILE, np.array(all_status))
        
        print("\n--- Processing Complete ---")
        print(f"Features shape: {np.array(all_features).shape} (Should be N x 22)")
        print(f"Saved to {PROCESSED_DATA_DIR}")
    else:
        print("\n--- Error: No data processed. Check paths and files. ---")

if __name__ == "__main__":
    process_data()