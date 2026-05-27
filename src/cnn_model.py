import torch
import torch.nn as nn

# --- CNN Architecture Hyperparameters ---
CONV_FILTERS = [32, 64, 128]
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE_UNITS = 64
DROPOUT_RATE = 0.3

class DeepNoiseCNN(nn.Module):
    """
    A lightweight 2D Convolutional Neural Network (CNN) in native PyTorch.
    Designed for classification of Mel-spectrogram inputs.
    
    Structure:
    - Input Permutation: Transposes channels-last shape (N, 128, 173, 1) to PyTorch channels-first layout (N, 1, 128, 173).
    - Block 1: Conv2D (32 filters) -> BatchNorm -> ReLU -> MaxPool2D (2x2)
    - Block 2: Conv2D (64 filters) -> BatchNorm -> ReLU -> MaxPool2D (2x2)
    - Block 3: Conv2D (128 filters) -> BatchNorm -> ReLU -> MaxPool2D (2x2)
    - Global Average Pooling: Reducings spatial size to channel dimensions.
    - Dense Block: Linear (64 units) -> ReLU -> Dropout (0.3)
    - Output Layer: Linear (4 units) -> Raw logits (compatible with nn.CrossEntropyLoss)
    """
    def __init__(self, num_classes=4):
        super(DeepNoiseCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_FILTERS[0], kernel_size=KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV_FILTERS[0])
        self.pool1 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(in_channels=CONV_FILTERS[0], out_channels=CONV_FILTERS[1], kernel_size=KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV_FILTERS[1])
        self.pool2 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(in_channels=CONV_FILTERS[1], out_channels=CONV_FILTERS[2], kernel_size=KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV_FILTERS[2])
        self.pool3 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dense Block
        self.fc1 = nn.Linear(CONV_FILTERS[2], DENSE_UNITS)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        
        # Output Classification Layer (Logits output)
        self.fc2 = nn.Linear(DENSE_UNITS, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Transpose input from channels_last (N, H, W, C) to PyTorch channels_first (N, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
            
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def build_cnn_model(num_classes=4):
    """
    Returns an instance of the native PyTorch DeepNoiseCNN model.
    Provided for compatibility with the build signature.
    """
    return DeepNoiseCNN(num_classes=num_classes)

if __name__ == "__main__":
    # Test model instantiation and display layer specifications
    model = build_cnn_model()
    print(model)
    
    # Test shape computation
    test_tensor = torch.randn(16, 128, 173, 1)
    outputs = model(test_tensor)
    print("Test Output Shape (Batch, Num Classes):", outputs.shape)
