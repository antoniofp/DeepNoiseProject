import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

# Append source directory to path to resolve local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, RESULTS_DIR, BATCH_SIZE, DEVICE, CLASSES
from cnn_model import build_cnn_model

def main():
    # 1. Load the unseen test dataset split
    test_data_path = os.path.join(MODELS_DIR, "test_split.npz")
    if not os.path.exists(test_data_path):
        print(f"Error: Test split file not found at {test_data_path}")
        print("Please run train.py first to generate the model and dataset splits.")
        sys.exit(1)
        
    print(f"Loading test split from: {test_data_path}")
    test_data = np.load(test_data_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    print(f"Test Set Shapes: Features = {X_test.shape} | Labels = {y_test.shape}")
    
    num_classes = len(CLASSES)
    print("Classes mapped to labels:", {name: idx for idx, name in enumerate(CLASSES)})
    
    # 2. Create DataLoader for memory-safe batched evaluation
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load the best saved model state dict
    best_model_path = os.path.join(MODELS_DIR, "best_cnn.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Model checkpoint not found at {best_model_path}")
        sys.exit(1)
        
    print(f"Instantiating model and loading weights from: {best_model_path}")
    model = build_cnn_model(num_classes=num_classes)
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    # 4. Perform prediction pass
    all_preds = []
    all_trues = []
    
    print(f"Running inference on {DEVICE}...")
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y_batch.numpy())
            
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
    # 5. Compute and print classification metrics
    print("\n" + "="*50)
    print("               CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_trues, all_preds, target_names=CLASSES)
    print(report)
    print("="*50)
    
    # 6. Generate and save the Confusion Matrix heatmap
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_trues, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=CLASSES, 
        yticklabels=CLASSES,
        cbar=False
    )
    plt.title("CNN Model: Animal Classification Test Set Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
    plt.ylabel("True Labels", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"Confusion Matrix saved to: {cm_path}")
    plt.close()

if __name__ == "__main__":
    main()
