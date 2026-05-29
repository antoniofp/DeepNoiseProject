# DeepNoise Project: Machine Learning Pipeline

## Overview
To build a robust acoustic event classification system, we must establish a clear end-to-end Machine Learning pipeline. This ensures that raw data is systematically transformed, models are trained fairly, and performance is evaluated accurately.

## Pipeline Architecture

### 1. Audio Acquisition (Audio Files)
**What we do:** Gather raw `.wav` files from the MIMII dataset and Freesound, organizing them into directories based on their respective classes.
**Why:** To establish a raw dataset ground truth before any manipulation occurs.

### 2. Preprocessing
**What we do:** Convert all audio to mono (single channel) and resample them to a uniform 22,050 Hz. We also normalize the amplitude (volume) to a standard level.
**Why:** Machine learning models require consistent input dimensions and scales. Normalization prevents the model from classifying sounds based solely on how loud the recording was.

### 3. Duration Normalization (Segmentation)
**What we do:** Ensure every audio clip is exactly 4.0 seconds long. Shorter clips are zero-padded (silence added to the end), and longer clips are truncated or split into multiple 4-second windows.
**Why:** Convolutional Neural Networks (CNNs) require fixed-size inputs. Consistent duration guarantees consistent image sizes in the next step.

### 4. Feature Extraction (Spectrogram Generation)
**What we do:** Transform the 1D time-domain audio signals into 2D Mel-spectrograms. We will save these representations as numerical arrays (tensors) or images.
**Why:** As decided in the Audio Representation Strategy, 2D Mel-spectrograms allow us to leverage powerful image-classification CNNs to detect acoustic patterns like mechanical friction or impacts.

### 5. Train / Validation / Test Split
**What we do:** Since the DCASE dataset is originally structured for unsupervised anomaly detection (putting only normal sounds in `train` and both normal/anomalous sounds in `test`), we first merge all files and group them into 8 clean, supervised folders (e.g., `fan_normal`, `fan_anomaly`, etc.) based on their filenames. Then, we divide this unified dataset into three subsets: 70% for Training, 15% for Validation, and 15% for Testing, using stratified sampling.
**Why:** Merging the folders ensures our training set contains both healthy and faulty machine sounds (supervised learning). Stratification ensures the class proportions (e.g., the ratio of normal-to-faulty fan sounds) remain identical across the Train, Validation, and Test sets, preventing evaluation bias.

### 6. Baseline Model
**What we do:** Train a simple, traditional classifier (e.g., Random Forest on flattened spectrograms or a basic Multi-Layer Perceptron). 
**Why:** To establish a minimum performance threshold. If our complex CNN cannot beat this simple model, the architecture or data needs fixing.

### 7. CNN Model Training
**What we do:** Train a 2D Convolutional Neural Network (such as a custom lightweight CNN or a pre-trained ResNet adapted for single-channel images) on the Mel-spectrograms.
**Why:** CNNs are the state-of-the-art for pattern recognition in 2D arrays (like our time-frequency acoustic images), capable of learning hierarchical spatial hierarchies (edges, textures, shapes).

### 8. Evaluation
**What we do:** Test the trained CNN on the 15% Test Split. We will calculate metrics like Accuracy, Precision, Recall, and the F1-Score, and plot a Confusion Matrix.
**Why:** To empirically prove how well the system detects normal operation versus mechanical faults.

### 9. Error Analysis
**What we do:** Analyze the False Positives and False Negatives from the confusion matrix. Listen to the specific audio clips the model got wrong.
**Why:** To understand the model's blind spots. For example, if it confuses "Imbalance" with "Factory Noise," we might need better data augmentation or a different Mel-band resolution in future iterations.
