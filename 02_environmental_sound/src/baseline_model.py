import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Append source directory to path to import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_DIR, CLASSES

def load_dataset():
    """
    Scans the data/features/ directory, loads the Log-Mel Spectrogram .npy files,
    extracts the fold number from the filename, and computes statistical features
    (mean and standard deviation across time frames for each Mel band).
    
    This reduces the feature representation from (128, 216) = 27,648 dimensions
    to a highly robust 256-dimensional vector (128 mean features + 128 std features),
    which prevents overfitting in traditional ML models.
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    print("Loading extracted Mel-spectrogram features...")
    
    # Check if features directory exists
    if not os.path.exists(FEATURES_DIR) or len(os.listdir(FEATURES_DIR)) == 0:
        raise FileNotFoundError("Features directory is empty. Please run preprocess.py first.")

    for class_name in CLASSES:
        class_dir = os.path.join(FEATURES_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_idx = CLASSES.index(class_name)
        npy_files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
        
        for npy_file in npy_files:
            # Filenames are of the format: {fold}-{clip_id}-{take}-{target}_{aug_name}.npy
            # The first character is the predefined fold index (1 to 5)
            fold = int(npy_file[0])
            
            # Load the 2D Mel-spectrogram of shape (128, 216)
            file_path = os.path.join(class_dir, npy_file)
            mel_spec = np.load(file_path)
            
            # Compute mean and standard deviation along the time axis (axis=1)
            mean_features = np.mean(mel_spec, axis=1) # Shape: (128,)
            std_features = np.std(mel_spec, axis=1)   # Shape: (128,)
            
            # Concatenate to form a 256-dim feature vector
            feature_vector = np.concatenate([mean_features, std_features])
            
            # Sort into splits based on the predefined fold
            if fold in [1, 2, 3]:
                X_train.append(feature_vector)
                y_train.append(class_idx)
            elif fold == 4:
                X_val.append(feature_vector)
                y_val.append(class_idx)
            elif fold == 5:
                X_test.append(feature_vector)
                y_test.append(class_idx)

    # Convert to NumPy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print(f"Dataset loaded:")
    print(f"  Training (Folds 1-3): {X_train.shape[0]} samples")
    print(f"  Validation (Fold 4):  {X_val.shape[0]} samples")
    print(f"  Testing (Fold 5):     {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return

    # 1. Instantiate and train Random Forest baseline
    print("\nTraining Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    
    # 2. Instantiate and train SVM baseline
    print("Training Support Vector Machine (SVM) Classifier...")
    svm_clf = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_clf.fit(X_train, y_train)

    # 3. Evaluate models on validation fold to select the best baseline
    rf_val_preds = rf_clf.predict(X_val)
    svm_val_preds = svm_clf.predict(X_val)

    rf_val_acc = accuracy_score(y_val, rf_val_preds)
    svm_val_acc = accuracy_score(y_val, svm_val_preds)

    print("\n=== Validation Performance Comparison (Fold 4) ===")
    print(f"Random Forest Validation Accuracy: {rf_val_acc * 100:.2f}%")
    print(f"Support Vector Machine Validation Accuracy: {svm_val_acc * 100:.2f}%")

    # Select the champion model
    if svm_val_acc >= rf_val_acc:
        best_clf = svm_clf
        model_name = "Support Vector Machine (SVM)"
    else:
        best_clf = rf_clf
        model_name = "Random Forest"

    print(f"\nWinner: {model_name} selected as the baseline model.")

    # 4. Evaluate the champion model on the unseen test set
    print(f"\n=== Final Evaluation of {model_name} on Unseen Test Split (Fold 5) ===")
    test_preds = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, test_preds, target_names=CLASSES))

if __name__ == "__main__":
    main()
