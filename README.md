# DeepNoise: Acoustic Monitoring for Predictive Maintenance

DeepNoise is an intelligent acoustic monitoring system designed to perform predictive maintenance on industrial machinery. By analyzing acoustic emission signals, the system classifies machine health (healthy vs. anomalous) to detect mechanical wear and faults before catastrophic failures occur.

This project is built using the DCASE 2020 Challenge Task 2 dataset (Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring).

---

## Project Architecture

The workspace is organized as follows:

```text
DeepNoiseProject/
├── data/                  # Local dataset files (git-ignored)
│   └── raw/               # Raw audio files (fan, pump, etc.)
├── docs/                  # Detailed documentation and design specifications
├── src/                   # Python source code and Jupyter notebooks
│   ├── .venv/             # Isolated Python virtual environment (git-ignored)
│   ├── config.py          # Centralized configuration and hyperparameters
│   ├── download_data.py   # Script to download/extract datasets from Zenodo
│   └── requirements.txt   # Python package dependencies
├── GEMINI.md              # Project mandates and developer guidelines
└── README.md              # This file
```

---

## Prerequisites

*   **Operating System:** Linux (configured for Nvidia CUDA acceleration)
*   **Python:** Version 3.12.x
*   **GPU Hardware:** Nvidia GPU (verified on GeForce RTX 3050 Ti)

---

## Getting Started

### 1. Set Up the Virtual Environment
Navigate to the project root and create an isolated Python virtual environment inside the `src/` directory:

```bash
python3 -m venv src/.venv
```

Activate the environment (optional if running scripts directly):

```bash
source src/.venv/bin/activate
```

### 2. Install Dependencies
Install the required packages. This includes PyTorch, Librosa (for audio preprocessing), and Scikit-learn (for baselines):

```bash
src/.venv/bin/pip install --upgrade pip
src/.venv/bin/pip install -r src/requirements.txt
```

### 3. Register Jupyter Kernel (Optional)
If you plan to run Jupyter Notebooks, register the virtual environment as a custom kernel to make it available to your Jupyter server:

```bash
src/.venv/bin/python3 -m ipykernel install --user --name=deepnoise --display-name="Python (DeepNoise)"
```

*Note: If you use VS Code, the editor's Python extension will automatically detect the `src/.venv` interpreter and allow you to select it without global registration.*

### 4. Acquire the Dataset
The machine learning pipeline requires raw WAV files from DCASE 2020. Run the automated script to download and extract the dataset subsets (currently fan and pump, totaling ~2.4 GB):

```bash
src/.venv/bin/python3 src/download_data.py
```

This will download the archives from Zenodo, extract them directly into `data/raw/`, and clean up the temporary zip files automatically.

---

## Running the Code

*   **Configurations:** All audio parameters (sample rate, Mel-spectrogram coefficients) and training configurations (epochs, batch size) are centralized in `src/config.py`.
*   **Device Target:** The configuration script dynamically checks for CUDA support. If an Nvidia GPU and drivers are present, calculations will automatically run on the GPU.
