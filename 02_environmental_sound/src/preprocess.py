import os
import sys
import glob
import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Append source directory to path to import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_DIR, FEATURES_DIR, SAMPLE_RATE, AUDIO_DURATION, N_MELS, N_FFT, HOP_LENGTH, CLASSES

# Calculate target number of samples
TARGET_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

def ensure_length(y, target_len):
    """
    Pads with zeros if shorter than target_len, or truncates if longer.
    """
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        return y[:target_len]
    return y

def preprocess_single_file(args):
    """
    Worker function to process a single audio file and generate its augmented variants:
    1. Loads raw WAV audio at target sample rate (22050 Hz).
    2. Normalizes amplitude to [-1.0, 1.0].
    3. Generates 6 augmented variants:
       - original (no change)
       - pitch_up (+1.5 semitones)
       - pitch_down (-1.5 semitones)
       - speed_up (1.15x speed)
       - speed_down (0.85x speed)
       - time_shift (0.5s shift with wrap-around)
       - noise (additive white Gaussian noise)
    4. Extracts Log-Mel Spectrogram (dB) features for each variant.
    5. Saves each feature map as a binary NumPy (.npy) file in data/features/<class_name>/.
    """
    src_path, dest_dir, filename_no_ext = args
    
    # Ensure class features directory exists (safe in multiprocessing)
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        # Load audio (automatically resampled to config SAMPLE_RATE)
        y, sr = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
        
        # 1. Normalize amplitude
        max_val = np.max(np.abs(y))
        if max_val > 0.0:
            y = y / max_val
            
        # Standardize duration
        y = ensure_length(y, TARGET_SAMPLES)
        
        # 2. Define variations
        variations = {}
        variations["orig"] = y
        
        # Pitch Shifts
        variations["pitch_up"] = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=1.5)
        variations["pitch_down"] = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=-1.5)
        
        # Time Stretches (must pad/truncate output back to standard duration)
        variations["speed_up"] = ensure_length(librosa.effects.time_stretch(y, rate=1.15), TARGET_SAMPLES)
        variations["speed_down"] = ensure_length(librosa.effects.time_stretch(y, rate=0.85), TARGET_SAMPLES)
        
        # Time Shift (roll audio waveform by 0.5 seconds)
        shift_samples = int(0.5 * SAMPLE_RATE)
        variations["time_shift"] = np.roll(y, shift_samples)
        
        # Additive Noise
        variations["noise"] = y + np.random.normal(0, 0.005, len(y))
        
        # 3. Extract and save Mel-spectrogram for each variation
        for var_name, y_var in variations.items():
            mel_spec = librosa.feature.melspectrogram(
                y=y_var, 
                sr=SAMPLE_RATE, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                n_mels=N_MELS
            )
            # Logarithmic amplitude scaling with fixed reference for stability
            mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
            
            dest_path = os.path.join(dest_dir, f"{filename_no_ext}_{var_name}.npy")
            np.save(dest_path, mel_spec_db)
            
        return True, src_path, None
    except Exception as e:
        return False, src_path, str(e)

def collect_tasks():
    """
    Read metadata.csv in RAW_DATA_DIR to gather the files we need to preprocess,
    building arguments for the parallel workers.
    """
    tasks = []
    metadata_path = os.path.join(RAW_DATA_DIR, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        return tasks
        
    df = pd.read_csv(metadata_path)
    
    for idx, row in df.iterrows():
        wav_filename = row['filename']
        class_name = row['category']
        src_path = os.path.join(RAW_DATA_DIR, wav_filename)
        
        if os.path.exists(src_path):
            filename_no_ext, _ = os.path.splitext(wav_filename)
            dest_dir = os.path.join(FEATURES_DIR, class_name)
            tasks.append((src_path, dest_dir, filename_no_ext))
            
    return tasks

def main():
    print("Scanning raw audio files from metadata...")
    tasks = collect_tasks()
    total_files = len(tasks)
    print(f"Collected {total_files} raw files to process.")
    
    if total_files == 0:
        print("No raw metadata or audio files found. Please run download_data.py first.")
        return
        
    print(f"\nStarting parallel preprocessing and data augmentation...")
    print(f"This will extract {total_files * 7} total Mel-spectrogram features.")
    
    success_count = 0
    failure_count = 0
    
    # Process files in parallel to utilize multi-core CPU architecture
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(preprocess_single_file, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            success, path, error_msg = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1
                print(f"\n[ERROR] Failed to preprocess {os.path.basename(path)}: {error_msg}")
                
            # Log progress
            if i % 40 == 0 or i == total_files:
                print(f"Progress: {i}/{total_files} files processed ({i*100/total_files:.1f}%) | "
                      f"Successes: {success_count} | Failures: {failure_count}")
                sys.stdout.flush()
                
    print(f"\nPreprocessing finished!")
    print(f"Successfully processed: {success_count} files (generating {success_count * 7} feature arrays).")
    print(f"Failures: {failure_count}.")

if __name__ == "__main__":
    main()
