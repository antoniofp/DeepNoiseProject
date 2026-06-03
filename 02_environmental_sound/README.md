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

### 2. Execution Pipeline
The entire workflow is automated and can be executed via the Jupyter Notebook [pipeline.ipynb](src/pipeline.ipynb), or step-by-step using Python scripts in `src/`:

*   **Ingest Data:** Download and extract the 8 animal classes from ESC-50:
    ```bash
    python src/download_data.py
    ```
*   **Preprocessing & Augmentations:** Downsample to 22.05kHz, apply 15 variations in parallel (1 original, 6 single, 8 combined), and extract Mel-spectrograms:
    ```bash
    python src/preprocess.py
    ```
*   **Traditional ML Baselines:** Train RF & SVM classifiers on statistical summaries, validating and testing on strictly clean splits:
    ```bash
    python src/baseline_model.py
    ```
*   **CNN Model Training & Evaluation:** Train both clean and augmented CNN models sequentially, and evaluate them on the unseen clean test split:
    ```bash
    python src/train.py
    python src/evaluate.py
    ```
*   **Train Production Model:** Train the final model across all 5 folds using a stratified 90/10 split (90% training with 15 augmentations, 10% clean validation for early stopping):
    ```bash
    python src/train_full.py
    ```

---

## Performance Summary

This project contains the **best-performing implementation** in the DeepNoise repository:
*   **Random Forest Baseline:** **54.69%** test accuracy.
*   **Clean CNN (No Augmentation):** **64.06%** test accuracy.
*   **Augmented CNN (15 Variations):** **70.31%** test accuracy.
*   **Full Production Model Validation:** **84.38%** clean validation accuracy.

For detailed analysis, architectures, and curves, see [docs/Final_Report.md](docs/Final_Report.md).

---

## Real-Time Live Demonstration

An interactive, real-time live demo is available to record and classify environmental sounds from your laptop microphone:

1.  Make sure your microphone is connected and configured in your OS settings.
2.  Run the live demo tool:
    ```bash
    python src/live_demo.py
    ```
3.  Choose **Option [3] Full Production Model** (Recommended).
4.  Press **Enter** to start the 5-second recording, play/make a sound, and review the predicted class probabilities!
