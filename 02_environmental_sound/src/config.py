import os

# Set PyTorch alloc configuration before importing torch to prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Base Directory: The directory of the 02_environmental_sound subproject.
# Calculated as the parent directory of the folder containing this config.py file.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Path Constants ---
# Raw audio files downloaded from ESC-50
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
# Processed / augmented audio files standardized in duration/format
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
# Extracted features (spectrograms stored as numpy .npy files)
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
# Model output directory to save weights and model architectures
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Directory for evaluation outputs (confusion matrices, reports)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure all relevant directories exist on disk
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Audio Preprocessing Hyperparameters ---
# Sample rate: 22050 Hz (standard downsampling for audio classification, balances detail and performance)
SAMPLE_RATE = 22050

# Duration: ESC-50 animal sound files are exactly 5.0 seconds.
AUDIO_DURATION = 5.0

# Mono: True, since single-channel is sufficient for animal classification
MONO = True

# --- Feature Extraction Hyperparameters (Mel-Spectrogram) ---
# n_mels: Number of Mel bands to generate. 128 is highly standard for environmental classification.
N_MELS = 128

# n_fft: Length of the FFT window. 2048 samples corresponds to ~93ms at 22050Hz.
N_FFT = 2048

# hop_length: Number of samples between successive FFT frames. 512 samples (~23ms)
# provides a solid temporal resolution while keeping spectrogram size manageable.
HOP_LENGTH = 512

# --- Model & Training Hyperparameters ---
# Number of epochs for training.
EPOCHS = 50

# Batch size: Fits easily within the 4GB VRAM limit of the RTX 3050 Ti.
BATCH_SIZE = 16

# Learning rate for the optimizer.
LEARNING_RATE = 0.001

# --- Hardware / Device Setup ---
# Prioritize NVIDIA RTX 3050 Ti (CUDA) for model execution, fallback to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target Animal Classes from ESC-50
CLASSES = ["dog", "rooster", "pig", "cow", "frog", "cat", "sheep", "hen"]
