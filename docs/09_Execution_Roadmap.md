# DeepNoise Project: Execution Roadmap

## The "Why": What Problem Does This Solve?
Before diving into the code, it is crucial to understand the real-world impact of this project:
- **Preventing Catastrophic Failures:** In factories, a broken machine can halt the entire production line. This system listens to the machines 24/7 to catch the earliest signs of wear (like a squeaking bearing) *before* the machine breaks.
- **Saving Money:** Unplanned downtime costs millions. By moving from "fix it when it breaks" to "fix it when it sounds weird," factories save massive amounts of money and resources.
- **Safety:** Detecting mechanical faults early prevents dangerous equipment failures that could harm human workers.

## My Role as an Autonomous AI Agent
In this project, I am not just a code-completion tool; I am functioning as your **Lead AI Engineer and Architect**. 
- **Autonomy:** My job is to execute this roadmap autonomously. I will write the scripts, set up the environment, and handle the data.
- **Problem Solving:** If a script fails or a model overfits, I will diagnose the error, adjust the code, and try again without needing you to write the fix.
- **Execution:** I will train the models, generate the graphs, and present the final results to you for review. Your role is the "Product Owner"—you define the goals and approve the direction, and I handle the technical execution.

---

## Phase 1: Environment Setup & Data Acquisition
*As the lead AI developer, this is my step-by-step technical plan for writing the code.*

### 1.1 Python Environment
- **Action:** Create a `requirements.txt` file and initialize a virtual environment (`venv`).
- **Dependencies:** Install core libraries: `librosa` (audio processing), `numpy`, `pandas`, `scikit-learn` (baseline model), `matplotlib`/`seaborn` (visualization), and `tensorflow` or `pytorch` (CNN model).

### 1.2 Data Downloading Script (Autonomous Execution)
- **Action:** I will write and execute a Python script (`src/download_data.py`) to automate the download process. It will programmatically download the 4 verified zip files (`dev_data_fan.zip`, `dev_data_pump.zip`, `dev_data_valve.zip`, `dev_data_slider.zip`) from Zenodo and extract them into the raw data directory.
- **Output:** Extracted directories: `data/raw/fan/`, `data/raw/pump/`, `data/raw/valve/`, and `data/raw/slider/`.

## Phase 2: Preprocessing & Feature Engineering

### 2.1 Audio Restructuring & Standardization Script
- **Action:** Write `src/preprocess.py`.
- **Logic:** 
  - Scan the extracted DCASE directories (`fan`, `pump`, `valve`, `slider`).
  - Read filenames from both DCASE `train/` and `test/` folders.
  - Parse the labels from the filenames (e.g. files containing `normal` are grouped into healthy classes; files containing `anomaly` are grouped into fault classes).
  - Loop through all identified `.wav` files.
  - Standardize formats: load using `librosa.load(sr=22050, mono=True)`.
  - Normalize amplitude.
  - Pad with zeros (silence) if shorter than 4 seconds, or truncate if longer.
- **Output:** Processed, structured audio saved in `data/processed/{class_name}/*.wav` (where class name matches our 8 target classes).

### 2.2 Spectrogram Generation Script
- **Action:** Write `src/extract_features.py`.
- **Logic:** 
  - Iterate through processed audio.
  - Apply `librosa.feature.melspectrogram()` to generate the 2D arrays.
  - Convert power to decibels using `librosa.power_to_db()`.
  - Save the arrays as `.npy` (NumPy arrays) to speed up training, storing them in `data/features/{class_name}/*.npy`.

## Phase 3: Baseline Implementation

### 3.1 Random Forest Script
- **Action:** Write `src/baseline_model.py`.
- **Logic:**
  - Load the `.npy` spectrograms.
  - Flatten them by averaging across the time axis to create a 1D vector per sample.
  - Split data using `sklearn.model_selection.train_test_split`.
  - Train `RandomForestClassifier`.
  - Print accuracy and the confusion matrix to the terminal.

## Phase 4: CNN Development

### 4.1 Custom CNN Architecture
- **Action:** Write `src/cnn_model.py`.
- **Logic:**
  - Define the architecture using Keras/PyTorch (Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten -> Dense -> Dropout -> Dense(Softmax)).
  - Compile with `CategoricalCrossentropy` and the `Adam` optimizer.

### 4.2 Training Loop
- **Action:** Write `src/train.py`.
- **Logic:**
  - Load the 2D `.npy` features and one-hot encode the labels.
  - Implement a `tf.data.Dataset` or PyTorch `DataLoader` for efficient batching.
  - Set up `EarlyStopping` (to prevent overfitting).
  - Train the model, saving the best weights to `models/best_cnn.h5`.

## Phase 5: Evaluation & Reporting

### 5.1 Validation Script
- **Action:** Write `src/evaluate.py`.
- **Logic:**
  - Load `models/best_cnn.h5` and the unseen Test Set.
  - Generate predictions.
  - Use `sklearn.metrics.classification_report` to compute Precision, Recall, and Macro F1-Score.
  - Use `seaborn` to generate a heatmap of the Confusion Matrix and save it as `results/confusion_matrix.png`.

### 5.2 Final Deliverable Generation
- **Action:** Consolidate all metrics and visualizations into a final `README.md` in the root of the repository, documenting how to run the codebase from scratch.
