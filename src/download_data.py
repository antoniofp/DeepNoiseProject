import os
import sys
import urllib.request
import zipfile
import time

# Import configuration paths from our centralized config module.
# Since config.py sets PYTORCH_CUDA_ALLOC_CONF, it is imported first.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_DIR

# --- Zenodo Dataset URLs ---
# Zenodo deposit ID for DCASE 2020 Challenge Task 2 Development Dataset is 3678171.
# The zip files are named dev_data_<class>.zip.
ZENODO_RECORD_URL = "https://zenodo.org/records/3678171/files/dev_data_{class_name}.zip?download=1"

# --- Target Subsets to Download ---
# To verify the pipeline efficiently without downloading the entire 4.6 GB dataset at once,
# we start with a subset: fan (1.4 GB) and pump (1.0 GB).
ACTIVE_CLASSES = ["fan", "pump"]

# --- Expected File Sizes ---
# Used for verification of download integrity.
EXPECTED_SIZES = {
    "fan": 1354774772,    # ~1.26 GiB (1.4 GB decimal)
    "pump": 1031279015    # ~0.96 GiB (1.0 GB decimal)
}


def download_progress_hook(count, block_size, total_size):
    """
    Custom progress reporting callback for urllib.request.urlretrieve.
    Displays percentage, downloaded size, and total size of the file.
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * 1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / total_size), 100)
    
    # Format and print the progress in place
    sys.stdout.write(
        f"\rDownloading... {percent}% | {progress_size / (1024 * 1024):.1f} MB / "
        f"{total_size / (1024 * 1024):.1f} MB | Speed: {speed} MB/s | Time: {duration:.1f}s"
    )
    sys.stdout.flush()


def download_and_extract():
    """
    Programmatically download active zip files from Zenodo,
    verify sizes, and extract them into data/raw/.
    """
    for class_name in ACTIVE_CLASSES:
        url = ZENODO_RECORD_URL.format(class_name=class_name)
        zip_path = os.path.join(RAW_DATA_DIR, f"dev_data_{class_name}.zip")
        
        print(f"\n=== Processing Class: {class_name.upper()} ===")
        
        # Check if the extracted directory already exists
        extracted_dir = os.path.join(RAW_DATA_DIR, class_name)
        if os.path.isdir(extracted_dir):
            print(f"Extracted directory already exists for {class_name} at: {extracted_dir}. Skipping.")
            continue
            
        # Download the zip file if it doesn't exist or is corrupted
        if os.path.isfile(zip_path):
            file_size = os.path.getsize(zip_path)
            if file_size == EXPECTED_SIZES[class_name]:
                print(f"Valid local zip found for {class_name} (size: {file_size / (1024 * 1024):.1f} MB). Skipping download.")
            else:
                print(f"Incomplete/corrupted zip found (size: {file_size} bytes). Re-downloading...")
                os.remove(zip_path)
                
        if not os.path.isfile(zip_path):
            print(f"Downloading from: {url}")
            try:
                urllib.request.urlretrieve(url, zip_path, download_progress_hook)
                print(f"\nDownload completed: {zip_path}")
            except Exception as e:
                print(f"\nError downloading {class_name}: {e}")
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
                continue
                
        # Verification check
        file_size = os.path.getsize(zip_path)
        if file_size != EXPECTED_SIZES[class_name]:
            print(f"WARNING: Verification failed! Size on disk: {file_size}, expected: {EXPECTED_SIZES[class_name]}")
            
        # Extraction
        print(f"Extracting {zip_path} into {RAW_DATA_DIR}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            print(f"Extraction successful for {class_name}.")
            # Clean up zip file to conserve storage space
            os.remove(zip_path)
            print(f"Removed zip file: {zip_path}")
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")


if __name__ == "__main__":
    download_and_extract()
