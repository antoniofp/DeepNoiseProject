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

There are two ways to execute the pipeline:

### Option A: Cell-by-Cell Notebook (Recommended for Humans)
Open the master notebook [pipeline.ipynb](file:///home/jantofp/Documents/DeepNoiseProject/src/pipeline.ipynb) in your notebook editor or VS Code. You can execute each step cell-by-cell. It uses Jupyter `%run` magics to execute the corresponding Python scripts sequentially while explaining each step.

### Option B: Command Line (CLI) Scripts
If you prefer running individual scripts via terminal commands, execute them in this exact order:

| Step | Command | Input | Output | Description |
|---|---|---|---|---|
| **1** | `src/.venv/bin/python3 src/download_data.py` | Remote Zenodo URLs | `data/raw/` | Downloads and extracts the DCASE dataset subsets (~2.4 GB). |
| **2** | `src/.venv/bin/python3 src/preprocess.py` | `data/raw/` | `data/processed/` | Standardizes audio length (4.0s), sample rate, and volume. |
| **3** | `src/.venv/bin/python3 src/extract_features.py` | `data/processed/` | `data/features/` | Extracts 2D Mel-spectrogram `.npy` arrays. |
| **4** | `src/.venv/bin/python3 src/train.py` | `data/features/` | `models/best_cnn.pth`<br>`models/test_split.npz` | Trains the PyTorch CNN on GPU and saves model & test sets. |
| **5** | `src/.venv/bin/python3 src/evaluate.py` | `models/test_split.npz` | `results/cnn_confusion_matrix.png` | Evaluates model and saves the confusion matrix heatmap. |

*   **Configurations:** All audio parameters (sample rate, Mel-spectrogram coefficients) and training configurations (epochs, batch size) are centralized in [config.py](file:///home/jantofp/Documents/DeepNoiseProject/src/config.py).
*   **Device Target:** The configuration script dynamically checks for CUDA support. If an Nvidia GPU (RTX 3050 Ti) and drivers are present, calculations will automatically run on the GPU.
