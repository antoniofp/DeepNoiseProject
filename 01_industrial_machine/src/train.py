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
    Scans the directories, pre-allocates a single contiguous NumPy array to match the total file count,
    and loads binary data directly into the allocated memory slices (bypassing slow Python lists).
    """
    print("Scanning feature directories...")
    
    # Identify directories representing classes (e.g., fan_normal, fan_anomaly, etc.)
    class_folders = sorted([d for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    
    # Create dynamic class-to-label mapping (e.g., {'fan_anomaly': 0, 'fan_normal': 1, ...})
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_folders)}
    print("Class-to-Label mapping:", class_to_label)
    
    # First, collect all file paths and their associated class labels to determine total count
    all_files_and_labels = []
    for class_name in class_folders:
        class_dir = os.path.join(FEATURES_DIR, class_name)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
        label = class_to_label[class_name]
        print(f"Found {len(npy_files)} files for class: {class_name}")
        for file_path in npy_files:
            all_files_and_labels.append((file_path, label))
            
    total_files = len(all_files_and_labels)
    if total_files == 0:
        print("Error: No feature files found. Run extract_features.py first.")
        sys.exit(1)
        
    # Read the first file to inspect its 2D dimensions dynamically (e.g. 128 Mel bands by 173 time frames)
    sample_spec = np.load(all_files_and_labels[0][0])
    height, width = sample_spec.shape
    print(f"Detected spectrogram dimensions: Height (Mel bands) = {height} | Width (Time bins) = {width}")
    print(f"Pre-allocating contiguous memory for {total_files} samples...")
    
    # Pre-allocate contiguous arrays directly in memory.
    # X shape will be (N, Height, Width, 1) to match channels-last standard grayscale image layout.
    X = np.empty((total_files, height, width, 1), dtype=np.float32)
    y = np.empty((total_files,), dtype=np.int64)
    
    # Load binary .npy files directly into their pre-allocated slots.
    # This completely bypasses Python list overhead and prevents multiple memory copies.
    for idx, (file_path, label) in enumerate(all_files_and_labels):
        X[idx, ..., 0] = np.load(file_path)
        y[idx] = label
        
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
            
            # --- The 5-Step Optimization Dance (Autograd and Pointer Mechanics) ---
            
            # Step 1: Clear the accumulated gradients from the last step.
            # PyTorch accumulates (sums) gradients by default. We must clear the old gradients 
            # (zeroing out the .grad attribute of every weight) so they don't corrupt the new step.
            optimizer.zero_grad()
            
            # Step 2: Feedforward pass - get predicted logits from the model.
            # When we pass x_batch through the model, PyTorch dynamically constructs a history tree
            # (computational graph). The output tensor `outputs` secretly holds pointers that trace
            # backward through every mathematical operation and point to the model's weights.
            outputs = model(x_batch)
            
            # Step 3: Compute the loss value (error penalty).
            # By passing `outputs` into the loss function, the calculated `loss` tensor is linked
            # to the top of the history tree, creating a complete map from the loss to the weights.
            loss = loss_fn(outputs, y_batch)
            
            # Step 4: Backward pass (Backpropagation).
            # PyTorch traverses the history tree backward starting from the `loss` tensor.
            # It calculates the gradient (direction of error) for each weight and writes that value
            # directly into the weight's own `.grad` attribute (e.g. self.conv1.weight.grad) in GPU memory.
            loss.backward()
            
            # Step 5: Update the weights in-place using the calculated gradients.
            # The optimizer holds a list of pointers to the model's weights. When we call step(),
            # it loops through those weights, reads their `.grad` attributes (which were just written
            # by loss.backward()), calculates the updates, and modifies the weights in-place.
            optimizer.step()
            
            # --- Track Training Metrics ---
            # 1. `loss.item()` represents the AVERAGE loss (error) per sample in the batch.
            #    We multiply it by the current batch size `len(x_batch)` to get the SUM of all errors in this batch.
            #    We do this because the last batch of the dataset might be smaller than 16 (e.g. 12 samples),
            #    and we must weight the errors correctly to calculate a true overall dataset average.
            train_loss += loss.item() * len(x_batch)
            
            # 2. `outputs` is shape (16, 4) holding raw scores (logits) for the 4 classes.
            #    `torch.argmax(outputs, dim=-1)` looks at each row (columns along dim=-1) and returns 
            #    the column index (0, 1, 2, or 3) containing the highest score (the model's prediction).
            preds = torch.argmax(outputs, dim=-1)
            
            # 3. `preds == y_batch` yields boolean values (True for match, False for mismatch).
            #    `torch.sum` converts True to 1 and False to 0, summing them up to count correct guesses.
            #    `.item()` extracts the raw Python integer from the single-value PyTorch tensor.
            train_correct += torch.sum(preds == y_batch).item()
            train_total += len(x_batch)
            
        # Calculate final training metrics for this epoch by dividing the total sum of errors 
        # and correct predictions by the total number of samples processed.
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # -------------------------------------------------------------------------
        # VALIDATION PASS (To evaluate generalization and prevent overfitting)
        # -------------------------------------------------------------------------
        # 1. Set the model to evaluation mode.
        #    This automatically disables layers like Dropout (we want all neurons active)
        #    and freezes BatchNorm layers so they use their accumulated "running averages"
        #    rather than computing new averages from the validation batch.
        model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 2. Open a context manager that disables gradient tracking history.
        #    Because we are not training, we do not need to calculate gradients or backward steps.
        #    Disabling history allows PyTorch to discard intermediate variables (activations)
        #    immediately, freeing up a massive amount of VRAM and speeding up calculations.
        with torch.no_grad():
            # Loop through validation batches in the loader
            for x_val, y_val in val_loader:
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(DEVICE)
                
                # Feed validation batch forward through model
                outputs = model(x_val)
                loss = loss_fn(outputs, y_val)
                
                # Track Validation Metrics (using the same math as the training metrics)
                val_loss += loss.item() * len(x_val)
                preds = torch.argmax(outputs, dim=-1)
                val_correct += torch.sum(preds == y_val).item()
                val_total += len(x_val)
                
        # Calculate final validation metrics for this epoch
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Print epoch metrics progress to trace model improvement
        print(f"Epoch {epoch+1:02d}/{EPOCHS} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
        
        # Record history for plotting curves later
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        # -------------------------------------------------------------------------
        # EARLY STOPPING AND CHECKPOINTING
        # -------------------------------------------------------------------------
        # If the validation loss in this epoch is the lowest we've seen so far, checkpoint the weights!
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reset patience counter since progress was made
            
            # Clone all model parameters.
            # In Python, model.state_dict() returns pointers to the weight tensors.
            # We must use `.clone()` to copy the actual float values to a new memory slot,
            # otherwise the weights in our best_state_dict would change as training continues.
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            
            # Save the copied weights to a binary file on disk
            torch.save(best_state_dict, best_model_path)
            print(f"  val_loss improved, saving model checkpoint to: {best_model_path}")
        else:
            # If validation loss did not improve, increment the patience penalty
            patience_counter += 1
            if patience_counter >= patience:
                # If we go 5 consecutive epochs without improvement, stop training early to avoid overfitting
                print(f"  Early stopping triggered! Training stopped at epoch {epoch+1}.")
                break
                
    # Restore the model weights from the best performing epoch.
    # This throws away the final overfitted weights and loads the optimal parameter states back in.
    model.load_state_dict(best_state_dict)
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
