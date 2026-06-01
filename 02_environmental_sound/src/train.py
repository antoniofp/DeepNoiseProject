import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Ensure VRAM allocation environment variables are set before PyTorch initializing.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Append source directory to path to resolve local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, CLASSES
from cnn_model import build_cnn_model

def load_dataset_by_fold():
    """
    Scans the extracted Mel-spectrogram features, parses the fold index from the
    filename prefix, and splits the data:
    - Folds 1-3: Training
    - Fold 4: Validation
    - Fold 5: Testing
    
    Returns features and labels pre-allocated into memory arrays.
    """
    print("Scanning feature directories for fold splits...")
    
    train_info = []
    val_info = []
    test_info = []
    
    # Check if features directory exists
    if not os.path.exists(FEATURES_DIR) or len(os.listdir(FEATURES_DIR)) == 0:
        raise FileNotFoundError("Features directory is empty. Run preprocess.py first.")

    for class_name in CLASSES:
        class_dir = os.path.join(FEATURES_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_idx = CLASSES.index(class_name)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
        
        for file_path in npy_files:
            basename = os.path.basename(file_path)
            # The first character of ESC-50 filenames represents the fold index (1 to 5)
            fold = int(basename[0])
            
            if fold in [1, 2, 3]:
                train_info.append((file_path, class_idx))
            elif fold == 4:
                val_info.append((file_path, class_idx))
            elif fold == 5:
                test_info.append((file_path, class_idx))

    # Read a sample to inspect input dimensions dynamically
    sample_spec = np.load(train_info[0][0])
    height, width = sample_spec.shape
    print(f"Spectrogram shape: {height} Mel bands x {width} time frames")
    
    def allocate_and_load(file_list):
        num_samples = len(file_list)
        X = np.empty((num_samples, height, width, 1), dtype=np.float32)
        y = np.empty((num_samples,), dtype=np.int64)
        for idx, (path, label) in enumerate(file_list):
            X[idx, ..., 0] = np.load(path)
            y[idx] = label
        return X, y

    print(f"Loading {len(train_info)} training samples (Folds 1-3)...")
    X_train, y_train = allocate_and_load(train_info)
    
    print(f"Loading {len(val_info)} validation samples (Fold 4)...")
    X_val, y_val = allocate_and_load(val_info)
    
    print(f"Loading {len(test_info)} test samples (Fold 5)...")
    X_test, y_test = allocate_and_load(test_info)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    # -------------------------------------------------------------------------
    # 1. LOAD DATASET BY FOLD
    # -------------------------------------------------------------------------
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_by_fold()
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. WRAP IN PYTORCH DATALOADERS
    # -------------------------------------------------------------------------
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # -------------------------------------------------------------------------
    # 3. INITIALIZE THE MODEL, OPTIMIZER, AND LOSS FUNCTION
    # -------------------------------------------------------------------------
    print(f"\nBuilding custom DeepNoiseCNN targeting device: {DEVICE}")
    model = build_cnn_model(num_classes=8)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "best_cnn.pth")
    
    # -------------------------------------------------------------------------
    # 4. TRAINING LOOP WITH EARLY STOPPING
    # -------------------------------------------------------------------------
    print(f"\nStarting CNN training (Epochs = {EPOCHS}, Batch Size = {BATCH_SIZE})...")
    
    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(x_batch)
            preds = torch.argmax(outputs, dim=-1)
            train_correct += torch.sum(preds == y_batch).item()
            train_total += len(x_batch)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(DEVICE)
                
                outputs = model(x_val)
                loss = loss_fn(outputs, y_val)
                
                val_loss += loss.item() * len(x_val)
                preds = torch.argmax(outputs, dim=-1)
                val_correct += torch.sum(preds == y_val).item()
                val_total += len(x_val)
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
        
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        # Checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, best_model_path)
            print(f"  val_loss improved, saving model checkpoint to: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered! Training stopped at epoch {epoch+1}.")
                break
                
    # Restore and save the champion weights
    model.load_state_dict(best_state_dict)
    torch.save(best_state_dict, best_model_path)
    print("\nModel training completed and best weights saved!")
    
    # Save the test set split to an .npz archive for evaluate.py
    test_data_path = os.path.join(MODELS_DIR, "test_split.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"Saved test set split to: {test_data_path}")
    
    # -------------------------------------------------------------------------
    # 5. PLOT LEARNING CURVES
    # -------------------------------------------------------------------------
    print("\nGenerating training learning curves...")
    plt.figure(figsize=(14, 5))
    
    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="tomato")
    plt.title("CNN Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train Acc", color="royalblue")
    plt.plot(history["val_accuracy"], label="Val Acc", color="tomato")
    plt.title("CNN Training vs. Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, "cnn_training_history.png")
    plt.savefig(history_path, dpi=300)
    print(f"Saved training history curves to: {history_path}")
    plt.close()

if __name__ == "__main__":
    main()
