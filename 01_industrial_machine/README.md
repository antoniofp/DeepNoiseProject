# DeepNoise: Acoustic Monitoring for Predictive Maintenance

DeepNoise is an intelligent acoustic monitoring system designed to perform predictive maintenance on industrial machinery. By analyzing acoustic emission signals, the system classifies machine health (healthy vs. anomalous) to detect mechanical wear and faults before catastrophic failures occur.

This project is built using the DCASE 2020 Challenge Task 2 dataset (Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring).

---

## Project Architecture

The workspace is organized as follows:

```text
01_industrial_machine/
├── data/                  # Local dataset files (git-ignored)
│   ├── raw/               # Raw audio files (fan, pump, etc.)
│   ├── processed/         # Resampled & normalized audio files
│   └── features/          # Extracted 2D Mel-spectrogram .npy files
├── docs/                  # Detailed documentation and design specifications
├── models/                # Saved model weights (best_cnn.pth, test_split.npz)
├── results/               # Evaluation artifacts (confusion matrix heatmaps)
└── src/                   # Python source code and Jupyter notebooks
    ├── config.py          # Centralized configuration and hyperparameters
    ├── download_data.py   # Script to download/extract datasets from Zenodo
    ├── preprocess.py      # Resamples and standardizes audio clips
    ├── extract_features.py# Extracts Mel-spectrogram arrays from audio
    ├── train.py           # Trains PyTorch CNN model on GPU
    ├── evaluate.py        # Runs inference on test data and evaluates results
    ├── baseline_model.ipynb # Random Forest baseline implementation
    ├── pipeline.ipynb     # Interactive master pipeline notebook
    └── requirements.txt   # Python package dependencies (shared with root)
```

---

## Prerequisites

*   **Operating System:** Linux (configured for Nvidia CUDA acceleration)
*   **Python:** Version 3.12.x
*   **GPU Hardware:** Nvidia GPU (verified on GeForce RTX 3050 Ti)

---

## Getting Started

### 1. Acquire the Dataset
The machine learning pipeline requires raw WAV files from DCASE 2020. Run the automated script using the shared virtual environment `.venv` at the repository root to download and extract the dataset subsets (currently fan and pump, totaling ~2.4 GB):

```bash
../../.venv/bin/python3 download_data.py
```
*(Run from the `01_industrial_machine/src/` folder)*

This will download the archives from Zenodo, extract them directly into `data/raw/`, and clean up the temporary zip files automatically.

---

## Running the Code

### Option A: Cell-by-Cell Notebook (Recommended for Humans)
Open the master notebook [pipeline.ipynb](file:///home/jantofp/Documents/DeepNoiseProject/01_industrial_machine/src/pipeline.ipynb) in your notebook editor or VS Code. You can execute each step cell-by-cell. It uses Jupyter `%run` magics to execute the corresponding Python scripts sequentially while explaining each step. Make sure to select the shared `.venv` kernel.

### Option B: Command Line (CLI) Scripts
If you prefer running individual scripts via terminal commands, navigate to `01_industrial_machine/src/` and execute them in this exact order:

| Step | Command | Input | Output | Description |
|---|---|---|---|---|
| **1** | `../../.venv/bin/python3 download_data.py` | Remote Zenodo URLs | `../data/raw/` | Downloads and extracts the DCASE dataset subsets (~2.4 GB). |
| **2** | `../../.venv/bin/python3 preprocess.py` | `../data/raw/` | `../data/processed/` | Standardizes audio length (4.0s), sample rate, and volume. |
| **3** | `../../.venv/bin/python3 extract_features.py` | `../data/processed/` | `../data/features/` | Extracts 2D Mel-spectrogram `.npy` arrays. |
| **4** | `../../.venv/bin/python3 train.py` | `../data/features/` | `../models/best_cnn.pth`<br>`../models/test_split.npz` | Trains the PyTorch CNN on GPU and saves model & test sets. |
| **5** | `../../.venv/bin/python3 evaluate.py` | `../models/test_split.npz` | `../results/cnn_confusion_matrix.png` | Evaluates model and saves the confusion matrix heatmap. |

*   **Configurations:** All audio parameters (sample rate, Mel-spectrogram coefficients) and training configurations (epochs, batch size) are centralized in [config.py](file:///home/jantofp/Documents/DeepNoiseProject/01_industrial_machine/src/config.py).
*   **Device Target:** The configuration script dynamically checks for CUDA support. If an Nvidia GPU (RTX 3050 Ti) and drivers are present, calculations will automatically run on the GPU.
