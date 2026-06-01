import os
import zipfile
import urllib.request
import pandas as pd
import io
from config import RAW_DATA_DIR, CLASSES

def download_and_extract_esc50():
    """
    Downloads the ESC-50 dataset from GitHub, parses the metadata CSV,
    and extracts ONLY the audio files and metadata for the 8 chosen animal classes
    to save space and time.
    """
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = os.path.join(RAW_DATA_DIR, "esc50_master.zip")

    # 1. Download the ZIP file with progress display
    if not os.path.exists(zip_path):
        print(f"Downloading ESC-50 dataset from {url}...")
        print("This is a ~600MB file and may take a few minutes depending on your internet speed.")
        
        # Simple progress callback
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            print(f"\rDownloading: {percent:.2f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end="")

        urllib.request.urlretrieve(url, zip_path, progress_hook)
        print("\nDownload complete!")
    else:
        print("Dataset ZIP already exists. Skipping download.")

    # 2. Open ZIP and find the CSV metadata file
    print("Parsing ZIP archive contents...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        
        # Find the CSV file dynamically (handling potential branch/folder prefix variations)
        csv_in_zip = None
        for name in namelist:
            if name.endswith("esc50.csv"):
                csv_in_zip = name
                break

        if not csv_in_zip:
            raise FileNotFoundError("Could not locate esc50.csv inside the downloaded ZIP file.")

        print(f"Found metadata at: {csv_in_zip}")
        
        # Read the CSV directly from the zip
        with zip_ref.open(csv_in_zip) as f:
            df = pd.read_csv(f)

        # Filter for the 8 target animal classes
        print(f"Filtering dataset for target animal classes: {CLASSES}")
        filtered_df = df[df['category'].isin(CLASSES)].copy()
        
        # Verify number of files
        num_expected_files = len(filtered_df)
        print(f"Filtered to {num_expected_files} files (expected 320: 8 classes * 40 samples).")

        # Save the filtered metadata CSV to RAW_DATA_DIR
        filtered_csv_path = os.path.join(RAW_DATA_DIR, "metadata.csv")
        filtered_df.to_csv(filtered_csv_path, index=False)
        print(f"Saved filtered metadata CSV to: {filtered_csv_path}")

        # Extract only the relevant audio files
        print("Extracting target animal audio files...")
        extracted_count = 0
        
        # Build filename to ZIP path map for quick extraction lookup
        audio_zip_paths = {}
        for name in namelist:
            basename = os.path.basename(name)
            if basename.endswith(".wav"):
                audio_zip_paths[basename] = name

        # Extract files
        for idx, row in filtered_df.iterrows():
            wav_filename = row['filename']
            if wav_filename in audio_zip_paths:
                zip_member = audio_zip_paths[wav_filename]
                # Read the audio data from zip and write it to the local directory
                with zip_ref.open(zip_member) as source, open(os.path.join(RAW_DATA_DIR, wav_filename), 'wb') as target:
                    target.write(source.read())
                extracted_count += 1
                if extracted_count % 40 == 0:
                    print(f"Extracted {extracted_count}/{num_expected_files} files...")
            else:
                print(f"Warning: Could not find {wav_filename} inside zip archive.")

        print(f"Successfully extracted {extracted_count} audio files to: {RAW_DATA_DIR}")

    # 3. Clean up the downloaded ZIP to free space
    print("Cleaning up temporary ZIP archive...")
    os.remove(zip_path)
    print("Cleanup complete. Dataset preparation is finished!")

if __name__ == "__main__":
    download_and_extract_esc50()
