import os

# Set PyTorch alloc configuration before importing torch to prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Base Directory: The root directory of the DeepNoiseProject.
# Calculated as the parent directory of the folder containing this config.py file.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Path Constants ---
# Raw audio files downloaded from Zenodo
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
# Processed audio files standardized in duration/format
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
# Sample rate: 22050 Hz is the standard for general audio signal processing in Librosa,
# providing a good balance between audio quality and computational footprint.
SAMPLE_RATE = 22050

# Duration: DCASE 2020 machine sounds are 10-second clips, but to speed up computation
# and align with common classification baselines, we standardize to 4.0 seconds.
# We will pad shorter clips and truncate longer ones.
AUDIO_DURATION = 4.0

# Mono: True, since multichannel audio is not needed for single-sensor machine monitoring.
MONO = True

# --- Feature Extraction Hyperparameters (Mel-Spectrogram) ---
# n_mels: Number of Mel bands to generate. Increased to 256 for finer frequency details.
N_MELS = 256

# n_fft: Length of the FFT window. 2048 samples corresponds to ~93ms at 22050Hz,
# which is a standard window size for speech and acoustic machine classification.
N_FFT = 2048

# hop_length: Number of samples between successive FFT frames. Reduced to 256 samples (~12ms)
# to double the temporal resolution, capturing short transient clicks and squeaks.
HOP_LENGTH = 256

# --- Model & Training Hyperparameters ---
# Number of epochs for baseline/CNN training. 30 epochs is sufficient for the CNN
# to converge when paired with early stopping.
EPOCHS = 30

# Batch size: Reduced to 16 to fit within the 4GB VRAM limit of the RTX 3050 Ti.
BATCH_SIZE = 16

# Learning rate: 0.001 is the standard default for the Adam optimizer, allowing stable convergence.
LEARNING_RATE = 0.001

# --- Hardware / Device Setup ---
# Prioritize NVIDIA RTX 3050 Ti (CUDA) for model execution, fallback to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Machine Classes: The 4 machine classes defined in the roadmap
CLASSES = ["fan", "pump", "valve", "slider"]
