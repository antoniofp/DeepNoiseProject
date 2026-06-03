import os
import sys
import glob
import random
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

def load_stratified_dataset(val_ratio=0.1, random_seed=42):
    """
    Shuffles and splits the original ESC-50 animal recordings to prevent data leakage:
    1. Groups Mel-spectrogram features by their original recording clip (base clip name).
    2. Randomly selects 10% of base clips per class for validation, 90% for training.
    3. For validation clips, loads only the original unaugmented files (_orig.npy).
    4. For training clips, loads all 15 variations (original + augmented).
    """
    print(f"Scanning features directory: {FEATURES_DIR}")
    random.seed(random_seed)
    
    train_paths = []
    val_paths = []
    
    if not os.path.exists(FEATURES_DIR) or len(os.listdir(FEATURES_DIR)) == 0:
        raise FileNotFoundError("Features directory is empty. Run preprocess.py first.")

    for class_name in CLASSES:
        class_dir = os.path.join(FEATURES_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_idx = CLASSES.index(class_name)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
        
        # 1. Group files by their base recording name
        # Example base name: '1-100032-A-0' extracted from '1-100032-A-0_orig.npy'
        base_clips = set()
        for path in npy_files:
            basename = os.path.basename(path)
            base_name = basename.split("_")[0]
            base_clips.add(base_name)
            
        base_clips = sorted(list(base_clips))  # Ensure deterministic order before shuffling
        random.shuffle(base_clips)
        
        # 2. Split base clips (10% validation, 90% training)
        num_val = int(len(base_clips) * val_ratio)
        if num_val == 0:
            num_val = 1
            
        val_clips = set(base_clips[:num_val])
        train_clips = set(base_clips[num_val:])
        
        print(f"  Class '{class_name}': {len(train_clips)} train clips, {len(val_clips)} validation clips")
        
        # 3. Associate files with splits
        for path in npy_files:
            basename = os.path.basename(path)
            base_name = basename.split("_")[0]
            is_orig = basename.endswith("_orig.npy")
            
            if base_name in val_clips:
                # Validation split only takes original clean recordings
                if is_orig:
                    val_paths.append((path, class_idx))
            elif base_name in train_clips:
                # Training split takes original + all augmented variations
                train_paths.append((path, class_idx))

    # Read a sample to inspect input dimensions dynamically
    sample_spec = np.load(train_paths[0][0])
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

    print(f"Loading {len(train_paths)} training samples (augmented)...")
    X_train, y_train = allocate_and_load(train_paths)
    
    print(f"Loading {len(val_paths)} validation samples (clean)...")
    X_val, y_val = allocate_and_load(val_paths)
    
    return X_train, y_train, X_val, y_val

def main():
    # 1. LOAD DATASET
    try:
        X_train, y_train, X_val, y_val = load_stratified_dataset(val_ratio=0.1)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)

    # 2. WRAP IN PYTORCH DATALOADERS
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
    
    print(f"\n==========================================")
    print(f" TRAINING MODEL: FULL PRODUCTION MODEL")
    print(f"==========================================")
    
    # 3. INITIALIZE THE MODEL, OPTIMIZER, AND LOSS FUNCTION
    print(f"Building custom DeepNoiseCNN targeting device: {DEVICE}")
    model = build_cnn_model(num_classes=len(CLASSES))
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    best_model_path = os.path.join(MODELS_DIR, "best_cnn_full.pth")
    
    # 4. TRAINING LOOP WITH EARLY STOPPING
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
    print(f"\nProduction model training completed and weights saved to: {best_model_path}")
    
    # 5. PLOT LEARNING CURVES
    print(f"\nGenerating training curves for full model...")
    plt.figure(figsize=(14, 5))
    
    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="tomato")
    plt.title("Production Model: Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train Acc", color="royalblue")
    plt.plot(history["val_accuracy"], label="Val Acc", color="tomato")
    plt.title("Production Model: Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, "cnn_training_history_full.png")
    plt.savefig(history_path, dpi=300)
    print(f"Saved production learning curves to: {history_path}")
    plt.close()

if __name__ == "__main__":
    main()
