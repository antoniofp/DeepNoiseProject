import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader

# Ensure VRAM allocation environment variables are set before PyTorch initializing
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Append source directory to path to resolve local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
from cnn_model import build_cnn_model

def load_dataset_2d():
    """
    Loads 2D Mel-spectrogram features from disk: (N, 128, 173) -> (N, 128, 173, 1).
    """
    print("Scanning feature directories...")
    X = []
    y = []
    
    class_folders = sorted([d for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_folders)}
    print("Class-to-Label mapping:", class_to_label)
    
    for class_name in class_folders:
        class_dir = os.path.join(FEATURES_DIR, class_name)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
        label = class_to_label[class_name]
        
        print(f"Loading {len(npy_files)} files for class: {class_name}")
        for file_path in npy_files:
            mel_spec = np.load(file_path)
            X.append(mel_spec)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    
    # Expand dimensions for single-channel representation (128 Mel bands, 173 time frames, 1 channel)
    X = np.expand_dims(X, axis=-1)
    
    return X, y, class_to_label

def main():
    # 1. Load dataset
    X, y, class_to_label = load_dataset_2d()
    print(f"Loaded dataset shapes: Features = {X.shape} | Labels = {y.shape}")
    
    # 2. Compute class weights to address training data skew
    unique_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y
    )
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("\nClass Weights:")
    for class_name, idx in class_to_label.items():
        print(f"  Class '{class_name}' ({idx}): weight = {weights[idx]:.4f}")
        
    # 3. Stratified splits (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Wrap datasets in PyTorch DataLoader
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
    
    # 4. Instantiate model
    print(f"\nBuilding CNN model targeting device: {DEVICE}")
    model = build_cnn_model(num_classes=4)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Native CrossEntropyLoss handles weighted categorical crossentropy natively on raw logits
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "best_cnn.pth")
    
    print(f"\nStarting model training on GPU in native PyTorch (Epochs = {EPOCHS}, Batch Size = {BATCH_SIZE})...")
    
    best_val_loss = float("inf")
    patience = 5
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
        
        # Early Stopping & Checkpointing
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
                
    # Restore best weights and save final model
    model.load_state_dict(best_state_dict)
    torch.save(best_state_dict, best_model_path)
    print("\nModel training completed and best weights saved!")
    
    # Save test set array split for evaluation in Phase 5
    test_data_path = os.path.join(MODELS_DIR, "test_split.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"Saved test set split to: {test_data_path}")
    
    # 7. Plot learning curves
    print("\nGenerating training learning curves...")
    plt.figure(figsize=(14, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="tomato")
    plt.title("CNN Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
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
