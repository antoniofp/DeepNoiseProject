# DeepNoise Project: Proposed Neural Network Architecture

## Chosen Architecture: Convolutional Neural Network (CNN)

To process our 2D Mel-spectrograms, we propose a lightweight, custom 2D CNN architecture. This avoids the excessive computational overhead of deep architectures (like ResNet or VGG) while still extracting the necessary spatial and temporal features from the audio representations.

### Architecture Flow
1. **Input Layer:** Receives the 2D Mel-spectrogram.
2. **Conv2D + ReLU:** Extracts low-level features (e.g., horizontal lines indicating continuous frequency bands).
3. **MaxPooling2D:** Downsamples the spatial dimensions, reducing computation and providing translation invariance.
4. **Conv2D + ReLU:** Extracts higher-level features (e.g., specific combinations of frequencies over short time bursts).
5. **MaxPooling2D:** Further downsamples.
6. **Flatten:** Converts the 2D feature maps into a 1D vector.
7. **Dense + ReLU:** Fully connected layer to interpret the extracted features.
8. **Dropout (Optional but recommended):** Randomly drops neurons during training to prevent overfitting.
9. **Dense (Output Layer) + Softmax:** Final classification layer.

## Technical Specifications

- **Input Shape:** `(128, 173, 1)`.
  * **128 (Height):** The number of Mel frequency bands.
  * **173 (Width):** The number of temporal frames. This is mathematically derived from our audio properties: a 4.0-second clip at 22,050 Hz contains 88,200 samples. Dividing this by our `hop_length` of 512 yields $\approx 173$ time steps (including standard boundary padding).
  * **1 (Channel):** Single channel (grayscale), representing the energy intensity.
- **Number of Output Classes:** 8 (Fan Normal/Anomaly, Pump Normal/Anomaly, Valve Normal/Anomaly, Slider Normal/Anomaly).
- **Type of Output Activation:** `Softmax` (to output a probability distribution across the 8 mutually exclusive classes).
- **Loss Function:** `Categorical Crossentropy` (standard for multi-class classification problems).
- **Optimizer:** `Adam` (Adaptive Moment Estimation), starting with a default learning rate (e.g., 0.001) for stable and fast convergence.
- **Evaluation Metrics:** 
  - `Accuracy` (primary metric for overall correctness).
  - `Macro F1-Score` (crucial due to expected class imbalances, ensuring the model doesn't just guess the majority class).

## Why a CNN is Appropriate
A CNN treats the Mel-spectrogram as an image. Mechanical acoustic events have distinct visual signatures in a spectrogram:
- A continuous squeak (friction) appears as a bright horizontal line at high frequencies.
- A periodic thump (imbalance) appears as vertical bars spaced evenly over time.
CNNs use local receptive fields (convolutional kernels) to detect these exact structural patterns (edges, textures) regardless of exactly *when* they occur in the 4-second window (thanks to translation invariance provided by Pooling layers). The Baseline Random Forest model completely ignores this structural geometry.

## Possible Risks and Mitigation
1. **Overfitting:** The model might memorize the training audio instead of learning general patterns. *Mitigation:* Use Dropout layers, L2 Regularization, and Early Stopping.
2. **Insufficient Data & Class Imbalance:** We have fewer "Fault" samples than "Normal" samples. The model might become biased. *Mitigation:* Apply Data Augmentation to the minority classes (time-shifting, adding Gaussian noise, slight pitch shifting) and use class weights during training.
3. **Poor Audio Quality / Environmental Noise:** High factory noise might mask the mechanical faults. *Mitigation:* We may need to apply noise-reduction algorithms during preprocessing or ensure our dataset includes "noisy" versions of faults so the model learns to filter it out.

## Hardware Optimization and Architectural Iteration
Because the training will be executed on a dedicated GPU (NVIDIA RTX 3050 Ti), the model training time will be highly optimized (expected to take under 2–3 minutes for 30–50 epochs on our small dataset). 

This hardware advantage allows the team to treat the proposed CNN architecture as a flexible baseline. If initial validation results are unsatisfactory (e.g. low precision or slow learning), we will leverage fast retraining cycles to experimentally tune hyperparameters:
- Increasing/decreasing filter sizes (e.g., 32 to 64) and adding layers.
- Adjusting dropout rates (e.g., 0.25 to 0.5) to manage overfitting.
- Experimenting with learning rates and batch sizes to stabilize convergence.
