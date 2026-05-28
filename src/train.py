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

# Ensure VRAM allocation environment variables are set before PyTorch initializing.
# "expandable_segments:True" prevents PyTorch from pre-allocating large contiguous blocks of VRAM,
# avoiding fragmentation and keeping VRAM usage low on laptop GPUs like the RTX 3050 Ti.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Append source directory to path to resolve local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
from cnn_model import build_cnn_model

def load_dataset_2d():
    """
    Loads 2D Mel-spectrogram features from disk:
    Reads .npy files, maps folder names to integer labels, and reshapes arrays.
    """
    print("Scanning feature directories...")
    X = []
    y = []
    
    # Identify directories representing classes (e.g., fan_normal, fan_anomaly, etc.)
    class_folders = sorted([d for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    
    # Create dynamic class-to-label mapping (e.g., {'fan_anomaly': 0, 'fan_normal': 1, ...})
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_folders)}
    print("Class-to-Label mapping:", class_to_label)
    
    for class_name in class_folders:
        class_dir = os.path.join(FEATURES_DIR, class_name)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
        label = class_to_label[class_name]
        
        print(f"Loading {len(npy_files)} files for class: {class_name}")
        for file_path in npy_files:
            # Load numpy binary array
            mel_spec = np.load(file_path)
            X.append(mel_spec)
            y.append(label)
            
    # Convert the Python lists of loaded individual arrays into a single unified NumPy array.
    # We do this because standard Python lists are slow and cannot be split or manipulated
    # by Scikit-Learn. Converting to np.array stacks the individual (128, 173) arrays into
    # a unified 3D block of shape (N, 128, 173).
    X = np.array(X)
    y = np.array(y)
    
    # Expand dimensions to add the dummy channel axis at the end, shifting from 3D to 4D:
    # Converts shape (N, 128, 173) to channels-last shape (N, 128, 173, 1).
    # - Why channels-last? NumPy, Scikit-Learn, and visualization libraries expect color channels at the end.
    # - How does PyTorch handle this? Our model's forward() function detects the single channel at the end
    #   and swaps it internally using x.permute(0, 3, 1, 2) before running the convolutions on the GPU.
    X = np.expand_dims(X, axis=-1)
    
    return X, y, class_to_label

def main():
    # -------------------------------------------------------------------------
    # 1. LOAD DATASET
    # -------------------------------------------------------------------------
    X, y, class_to_label = load_dataset_2d()
    print(f"Loaded dataset shapes: Features = {X.shape} | Labels = {y.shape}")
    
    # -------------------------------------------------------------------------
    # 2. COMPUTE CLASS WEIGHTS (To handle dataset imbalance)
    # -------------------------------------------------------------------------
    # Because healthy sounds outnumber anomalies, we compute weights to penalize
    # errors on minority classes more severely.
    unique_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y
    )
    # Convert weights to a PyTorch float tensor and send it to the GPU/CPU device
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("\nClass Weights:")
    for class_name, idx in class_to_label.items():
        print(f"  Class '{class_name}' ({idx}): weight = {weights[idx]:.4f}")
        
    # -------------------------------------------------------------------------
    # 3. SPLIT DATASET (70% Train, 15% Validation, 15% Test)
    # -------------------------------------------------------------------------
    # First split: Separate 70% for training and 30% temporary pool.
    # stratify=y ensures all splits maintain the same class proportions.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Second split: Divide the 30% temporary pool equally into Validation (15%) and Test (15%).
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # -------------------------------------------------------------------------
    # 4. WRAP IN PYTORCH DATALOADERS (For batch management)
    # -------------------------------------------------------------------------
    # TensorDataset pairs the feature tensors with their corresponding labels.
    # DataLoader manages slicing the tensors into batch slices of size BATCH_SIZE (16).
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    # shuffle=True mixes training sample order on every epoch to prevent ordering bias.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    # Validation data does not need to be shuffled.
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # -------------------------------------------------------------------------
    # 5. INITIALIZE THE MODEL, OPTIMIZER, AND LOSS FUNCTION
    # -------------------------------------------------------------------------
    print(f"\nBuilding CNN model targeting device: {DEVICE}")
    
    # Instantiate the main model object
    model = build_cnn_model(num_classes=4)
    # Move the model weights to the target GPU (cuda) or CPU memory
    model.to(DEVICE)
    
    # Initialize the Adam Optimizer.
    # We pass it model.parameters() so it holds pointers to update the model weights in-place.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize the Loss Function (Categorical Cross Entropy).
    # We supply our calculated class weights so it applies the penalty weights during the loss calculation.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Ensure directory to save model exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "best_cnn.pth")
    
    print(f"\nStarting model training on GPU in native PyTorch (Epochs = {EPOCHS}, Batch Size = {BATCH_SIZE})...")
    
    # Variables to track early stopping checkpointing
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    
    # Initialize dictionary to save validation weights
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Log history of metrics over epochs for plotting
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    # -------------------------------------------------------------------------
    # 6. THE EPOCH TRAINING LOOP
    # -------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        # Set the model to training mode. 
        # This tells layers like Dropout to turn ON.
        model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Loop through the training batches provided by the DataLoader
        for x_batch, y_batch in train_loader:
            # Move the batch data to the target GPU or CPU
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # --- The 5-Step Optimization Dance ---
            
            # Step 1: Clear the accumulated gradients from the last step
            optimizer.zero_grad()
            
            # Step 2: Feedforward pass - get predicted logits from the model
            outputs = model(x_batch)
            
            # Step 3: Compute the loss value (error penalty)
            loss = loss_fn(outputs, y_batch)
            
            # Step 4: Backward pass - compute gradients of the loss w.r.t model weights
            loss.backward()
            
            # Step 5: Update the weights in-place using the calculated gradients
            optimizer.step()
            
            # --- Track Training Metrics ---
            # Multiply loss by batch size since loss is averaged over the batch
            train_loss += loss.item() * len(x_batch)
            # Find index of the highest logit prediction
            preds = torch.argmax(outputs, dim=-1)
            # Count correct predictions
            train_correct += torch.sum(preds == y_batch).item()
            train_total += len(x_batch)
            
        # Calculate final training metrics for this epoch
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # -------------------------------------------------------------------------
        # VALIDATION PASS (To evaluate generalization and prevent overfitting)
        # -------------------------------------------------------------------------
        # Set model to evaluation mode. 
        # This tells layers like Dropout to turn OFF.
        model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Turn off gradient calculation history to save memory and compute speed
        with torch.no_grad():
            # Loop through validation batches
            for x_val, y_val in val_loader:
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(DEVICE)
                
                # Feedforward pass
                outputs = model(x_val)
                # Compute loss
                loss = loss_fn(outputs, y_val)
                
                # Track Validation Metrics
                val_loss += loss.item() * len(x_val)
                preds = torch.argmax(outputs, dim=-1)
                val_correct += torch.sum(preds == y_val).item()
                val_total += len(x_val)
                
        # Calculate final validation metrics for this epoch
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Print epoch metrics progress
        print(f"Epoch {epoch+1:02d}/{EPOCHS} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
        
        # Record history for plotting
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        # -------------------------------------------------------------------------
        # EARLY STOPPING AND CHECKPOINTING
        # -------------------------------------------------------------------------
        # If the validation loss is the best we've seen, save the weights!
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Clone state dict parameters to prevent references from updating on subsequent epochs
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            # Write the best model weights to disk
            torch.save(best_state_dict, best_model_path)
            print(f"  val_loss improved, saving model checkpoint to: {best_model_path}")
        else:
            # If validation loss did not improve, increment early stopping counter
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered! Training stopped at epoch {epoch+1}.")
                break
                
    # Restore the model weights from the best performing epoch
    model.load_state_dict(best_state_dict)
    # Save the final best weights
    torch.save(best_state_dict, best_model_path)
    print("\nModel training completed and best weights saved!")
    
    # Save the unseen test set split to an .npz archive for evaluate.py
    test_data_path = os.path.join(MODELS_DIR, "test_split.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"Saved test set split to: {test_data_path}")
    
    # -------------------------------------------------------------------------
    # 7. PLOT LEARNING CURVES
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
    
    # Save learning history plot as image
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, "cnn_training_history.png")
    plt.savefig(history_path, dpi=300)
    print(f"Saved training history curves to: {history_path}")
    plt.close()

if __name__ == "__main__":
    main()
