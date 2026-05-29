import os
import sys
import glob
import librosa
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Append source directory to path to import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE, AUDIO_DURATION, MONO

# Target sample length calculated as Sample Rate * Duration (e.g., 22050 * 4.0 = 88200 samples)
TARGET_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)


def preprocess_single_file(args):
    """
    Worker function to process a single audio file:
    1. Loads the audio using the standardized sampling rate and mono channel settings.
    2. Normalizes the amplitude range to [-1.0, 1.0].
    3. Pads shorter clips with zeros or truncates longer clips to reach exactly 4.0 seconds.
    4. Writes the processed audio to the target directory.
    """
    src_path, dest_dir, filename = args
    dest_path = os.path.join(dest_dir, filename)
    
    # Ensure destination directory exists (safe for multiprocessing)
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        # Load audio file (Librosa handles resampling to the config SAMPLE_RATE automatically)
        y, sr = librosa.load(src_path, sr=SAMPLE_RATE, mono=MONO)
        
        # Normalize amplitude to prevent volume differences from biasing the network
        max_val = np.max(np.abs(y))
        if max_val > 0.0:
            y = y / max_val
            
        # Pad with silence (zeros) if shorter than target samples
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")
        # Truncate if longer than target samples
        elif len(y) > TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
            
        # Save processed waveform using soundfile (optimized C implementation)
        sf.write(dest_path, y, SAMPLE_RATE)
        return True, src_path, None
    except Exception as e:
        return False, src_path, str(e)


def collect_tasks():
    """
    Scan the raw dataset directory, parse DCASE filenames to resolve supervised labels
    (normal vs. anomaly), and prepare the list of files to process.
    """
    tasks = []
    
    # DCASE 2020 Task 2 structure contains 'fan' and 'pump' raw directories
    for machine_type in ["fan", "pump"]:
        machine_raw_dir = os.path.join(RAW_DATA_DIR, machine_type)
        if not os.path.isdir(machine_raw_dir):
            continue
            
        # Scan both train/ and test/ folders inside raw directory
        for split in ["train", "test"]:
            split_dir = os.path.join(machine_raw_dir, split)
            if not os.path.isdir(split_dir):
                continue
                
            wav_files = glob.glob(os.path.join(split_dir, "*.wav"))
            for src_path in wav_files:
                filename = os.path.basename(src_path)
                
                # Resolve supervised class name based on filename prefix (normal_* vs. anomaly_*)
                if filename.startswith("normal_"):
                    class_name = f"{machine_type}_normal"
                elif filename.startswith("anomaly_"):
                    class_name = f"{machine_type}_anomaly"
                else:
                    # Fallback in case of unexpected naming convention
                    class_name = f"{machine_type}_unknown"
                    
                # Prepend the split name ('train_' or 'test_') to the filename to avoid name collisions
                unique_filename = f"{split}_{filename}"
                dest_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
                tasks.append((src_path, dest_dir, unique_filename))
                
    return tasks


def main():
    print("Scanning raw dataset directories...")
    tasks = collect_tasks()
    total_files = len(tasks)
    print(f"Collected {total_files} audio files for processing.")
    
    if total_files == 0:
        print("No raw audio files found. Please run src/download_data.py first.")
        return
        
    print("\nStarting parallel audio preprocessing...")
    success_count = 0
    failure_count = 0
    
    # Execute preprocessing in parallel using ProcessPoolExecutor (one process per CPU core)
    # This leverages the 12 available CPU cores for up to 10x speedup.
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(preprocess_single_file, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            success, path, error_msg = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1
                print(f"\n[ERROR] Failed to process {os.path.basename(path)}: {error_msg}")
                
            # Log progress every 500 files to avoid terminal spam in background execution
            if i % 500 == 0 or i == total_files:
                print(f"Progress: {i}/{total_files} files processed ({i*100/total_files:.1f}%) | "
                      f"Successes: {success_count} | Failures: {failure_count}")
                sys.stdout.flush()
                
    print(f"\nPreprocessing finished! Successfully processed: {success_count} files. Failures: {failure_count}.")


if __name__ == "__main__":
    main()
