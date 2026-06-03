# DeepNoise: Acoustic Classification Monorepo

Welcome to the **DeepNoise** repository. This monorepo hosts two parallel machine learning projects focused on classifying acoustic data using traditional machine learning and Deep Learning Convolutional Neural Networks (CNNs).

---

## Monorepo Layout

The repository is organized into parallel project directories sharing a single, centralized Python virtual environment:

```text
DeepNoiseProject/
├── .venv/                   # Centrally shared Python virtual environment (git-ignored)
├── .gitignore               # Root git ignore patterns
├── README.md                # This file
│
├── 01_industrial_machine/   # Project 1: Machine sound classification (DCASE 2020)
│   ├── data/                # Raw/Processed dataset directories (git-ignored)
│   ├── docs/                # Design specifications and documentation
│   ├── models/              # Saved model checkpoints and weights
│   ├── results/             # Evaluation metrics and heatmaps
│   └── src/                 # CNN & Random Forest source code & notebooks
│
└── 02_environmental_sound/  # Project 2: Environmental acoustic classification
    ├── data/                # Raw/Processed dataset directories (git-ignored)
    ├── docs/                # Design specifications and documentation
    ├── models/              # Saved model checkpoints and weights
    ├── results/             # Evaluation metrics and heatmaps
    └── src/                 # Source code and notebooks
```

---

## Getting Started

### 1. Prerequisites
*   **Operating System:** Linux (configured for Nvidia CUDA acceleration)
*   **Python:** Version 3.12.x
*   **GPU Hardware:** Nvidia GPU (verified on GeForce RTX 3050 Ti)

### 2. Set Up the Shared Virtual Environment
From the repository root, create the shared Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install the dependencies using the requirements spec in Project 1 (which acts as the baseline environment for the monorepo):

```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r 01_industrial_machine/src/requirements.txt
```

### 3. Register Jupyter Kernel (Optional)
If running Jupyter notebooks from VS Code or a notebook server, register the kernel:

```bash
.venv/bin/python3 -m ipykernel install --user --name=deepnoise --display-name="Python (DeepNoise)"
```

---

## Subprojects Overview

### [01. Industrial Machine Acoustic Monitoring](file:///home/jantofp/Documents/DeepNoiseProject/01_industrial_machine/README.md)
*   **Objective:** Classify acoustic emission signals from industrial machines (fans, pumps, valves, sliders) to distinguish between healthy and anomalous states.
*   **Dataset:** DCASE 2020 Challenge Task 2.
*   **Models:** 128-dim Random Forest baseline, and a custom **4x4 Hybrid Pooling 2D CNN** (achieving 61% Recall on Fan anomaly and 60% Recall on Pump anomaly).
*   **Status:** Complete & Runnable.

### [02. Environmental Sound Classification (Animal Sounds)](file:///home/jantofp/Documents/DeepNoiseProject/02_environmental_sound/README.md)
*   **Objective:** Classify environmental animal vocalizations (8 target classes: dog, rooster, pig, cow, frog, cat, sheep, hen) from the ESC-50 dataset.
*   **Models:** 256-dim Random Forest baseline (**54.69%** accuracy), Clean CNN (**64.06%**), and an Augmented CNN (**70.31%** test accuracy; **84.38%** production validation accuracy). 
*   **Performance:** Represents the **best overall classification performance** in this repository (+15.62% test accuracy over its baseline).
*   **Status:** Complete & Production-Ready (includes real-time microphone live demonstration tool).
