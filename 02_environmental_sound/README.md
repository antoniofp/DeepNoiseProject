# Environmental Sound Classification System

This subproject implements an acoustic event classification pipeline designed to classify environmental sounds (e.g., urban noises, nature sounds, or custom environmental acoustic events). 

The pipeline processes raw audio waveforms, standardizes them, extracts robust 2D representations (such as Mel-spectrograms), and classifies them using both traditional baseline classifiers (such as Random Forest) and Deep Learning Convolutional Neural Networks (CNNs).

---

## Project Architecture

The workspace is organized as follows:

```text
02_environmental_sound/
├── data/                  # Environmental audio datasets (git-ignored)
│   ├── raw/               # Raw audio recordings organized by class
│   ├── processed/         # Standardized audio waveforms
│   └── features/          # Extracted 2D features (Mel-spectrograms)
├── docs/                  # Project specifications and acoustic analysis documentation
├── models/                # Saved models and checkpoint weights
├── results/               # Classification reports, confusion matrices, and training logs
└── src/                   # Python source code and Jupyter notebooks
```

---

## Setup & Execution

### 1. Requirements
This subproject shares the repository's root virtual environment `.venv`. To activate it:

```bash
source ../.venv/bin/activate
```

### 2. Coming Soon
The audio pipeline, feature extraction, and neural network classification architectures for this project are currently being initialized.
