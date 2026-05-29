# DeepNoise Project: Baseline Model Proposal

## Chosen Baseline Model
We will use a **Random Forest Classifier** as our baseline model.

## Input Features
While our final CNN will use full 2D Mel-spectrograms, feeding a raw, high-dimensional 2D image into a Random Forest is inefficient. Instead, the baseline model will receive **flattened, globally averaged Mel-spectrograms**, or alternatively, **MFCCs** (Mel-Frequency Cepstral Coefficients). For simplicity and speed in the baseline, we will extract the mean and standard deviation of the Mel-frequency bands across the time axis, resulting in a 1D vector of numerical features for each audio sample.

## Why it is a Reasonable Comparison
1. **Robustness:** Random Forests handle non-linear relationships well and are highly resistant to overfitting on tabular/1D data compared to simple Logistic Regression.
2. **Speed and Simplicity:** They require very little hyperparameter tuning to get a "good enough" result and train in seconds on standard CPUs.
3. **Contrast with CNNs:** A Random Forest taking averaged 1D features entirely ignores the *spatial and temporal relationships* present in the 2D spectrogram. If the CNN (which *does* see these relationships) significantly outperforms the Random Forest, it proves that the temporal/spatial patterns (like the specific rhythm of a machine impact) are crucial for solving the problem.

## Expected Results
We expect the Random Forest to perform decently well on distinguishing vastly different classes (like "Ambient Noise" vs. "Normal Operation"). However, we expect it to struggle (lower accuracy, higher confusion) when differentiating between similar mechanical states (e.g., "Normal" vs. "Imbalance"), because averaging the features over time destroys the rhythmic, periodic nature of the imbalance impact. 
Overall, we expect an accuracy of around **60-70%**. This sets a clear, achievable threshold that our CNN must beat (aiming for >85%) to justify its added complexity.
