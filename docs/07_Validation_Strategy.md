# DeepNoise Project: Validation Strategy

## Overview
Evaluating a Machine Learning model correctly is as important as building it. During the implementation stage, we will use a comprehensive validation strategy to ensure our CNN genuinely learns to distinguish acoustic events rather than memorizing the data or exploiting class imbalances.

## 1. Data Splitting Strategy
We will divide our dataset into three distinct sets using stratified sampling:
- **Training Set (70%):** Used strictly for the model to learn the weights and patterns.
- **Validation Set (15%):** Used during training to tune hyperparameters (like learning rate) and to monitor for overfitting.
- **Test Set (15%):** A completely hidden "hold-out" set used only once at the very end to evaluate the final model's real-world performance.

## 2. Evaluation Metrics
We will calculate the following metrics on the Test Set:

### Accuracy
The percentage of total correct predictions. While useful for a quick glance, **accuracy alone is heavily misleading in unbalanced datasets.** For example, if our dataset has 90% "Normal Operation" samples and 10% "Bearing Fault" samples, a model that simply guesses "Normal" every single time will achieve 90% accuracy, but it will be completely useless for detecting faults.

### Precision, Recall, and F1-Score
Because accuracy is insufficient, we will rely on:
- **Precision:** Out of all the times the model predicted a "Bearing Fault", how many were actually Bearing Faults? (Measures false positives).
- **Recall:** Out of all the actual "Bearing Faults" in the dataset, how many did the model successfully find? (Measures false negatives). In predictive maintenance, *Recall* is often the most important metric because missing a fault (False Negative) is much more expensive than a false alarm (False Positive).
- **F1-Score:** The harmonic mean of Precision and Recall. We will use the **Macro F1-Score** to give equal weight to all classes, ensuring the model performs well on rare faults, not just abundant normal data.

### 3. Confusion Matrix
A visual table comparing the *Predicted Classes* against the *Actual Classes*. This is critical for visualizing exactly *where* the model is making mistakes (e.g., distinguishing between "Imbalance" and "Bearing Fault", but never confusing them with "Ambient Noise").

### 4. Training and Validation Curves
We will plot Loss and Accuracy across training epochs. 
- If Training Loss goes down but Validation Loss goes up, the model is **overfitting** (memorizing).
- If both remain high, the model is **underfitting** (failing to learn).
These curves dictate when to trigger Early Stopping.

### 5. Error Analysis by Class
After evaluation, we will isolate the False Positives and False Negatives for each specific class. By listening to the misclassified audio clips and viewing their Mel-spectrograms, we can identify patterns in the errors (e.g., "The model always fails when high background noise is present"). This analysis will drive the next iteration of dataset cleaning or model tuning.
