import os
import sys
import glob
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Append source directory to path to import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DATA_DIR, FEATURES_DIR, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH


def extract_features_single_file(args):
    """
    Worker function to extract features for a single file:
    1. Loads the preprocessed, standardized waveform.
    2. Computes the 2D Mel-spectrogram using parameters defined in config.py.
    3. Converts the power representation to decibels (logarithmic scaling).
    4. Saves the resulting matrix as a binary NumPy array (.npy).
    """
    src_path, dest_dir, filename_no_ext = args
    dest_path = os.path.join(dest_dir, f"{filename_no_ext}.npy")
    
    # Ensure target subdirectory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        # Load preprocessed audio (standardized at config SAMPLE_RATE)
        # We specify sr=SAMPLE_RATE to prevent librosa from running redundant sample rate checks
        y, sr = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
        
        # Compute the Mel-spectrogram representation
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=SAMPLE_RATE, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            n_mels=N_MELS
        )
        
        # Convert power to decibels (dB) to capture human logarithmic perception of loudness
        # and normalize numerical ranges for neural network training
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save array as a compact binary .npy file for fast loading during training
        np.save(dest_path, mel_spec_db)
        return True, src_path, None
    except Exception as e:
        return False, src_path, str(e)


def collect_tasks():
    """
    Scan data/processed/ directories to collect the standardized WAV files
    and determine the corresponding output directories under data/features/.
    """
    tasks = []
    
    # Scan the processed directories (e.g., fan_normal, fan_anomaly, pump_normal, pump_anomaly)
    if not os.path.isdir(PROCESSED_DATA_DIR):
        return tasks
        
    class_dirs = [d for d in os.listdir(PROCESSED_DATA_DIR) if os.path.isdir(os.path.join(PROCESSED_DATA_DIR, d))]
    
    for class_name in class_dirs:
        class_processed_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
        wav_files = glob.glob(os.path.join(class_processed_dir, "*.wav"))
        
        dest_dir = os.path.join(FEATURES_DIR, class_name)
        for src_path in wav_files:
            filename = os.path.basename(src_path)
            filename_no_ext, _ = os.path.splitext(filename)
            tasks.append((src_path, dest_dir, filename_no_ext))
            
    return tasks


def main():
    print("Scanning processed audio directories...")
    tasks = collect_tasks()
    total_files = len(tasks)
    print(f"Collected {total_files} processed files for feature extraction.")
    
    if total_files == 0:
        print("No processed audio files found. Please run src/preprocess.py first.")
        return
        
    print("\nStarting parallel Mel-spectrogram extraction...")
    success_count = 0
    failure_count = 0
    
    # Execute feature extraction in parallel using all 12 CPU cores.
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_features_single_file, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            success, path, error_msg = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1
                print(f"\n[ERROR] Failed to extract features for {os.path.basename(path)}: {error_msg}")
                
            # Log progress every 500 files to avoid terminal spam
            if i % 500 == 0 or i == total_files:
                print(f"Progress: {i}/{total_files} features extracted ({i*100/total_files:.1f}%) | "
                      f"Successes: {success_count} | Failures: {failure_count}")
                sys.stdout.flush()
                
    print(f"\nFeature extraction finished! Successfully extracted: {success_count} arrays. Failures: {failure_count}.")


if __name__ == "__main__":
    main()
