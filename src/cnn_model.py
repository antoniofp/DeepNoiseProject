import torch
import torch.nn as nn

# --- CNN Architecture Hyperparameters ---
# The number of convolutional filters (feature maps) extracted in each respective block.
CONV_FILTERS = [32, 64, 128]

# The height and width of the 2D sliding filter window.
KERNEL_SIZE = 3

# The size of the Max Pooling window.
POOL_SIZE = 2

# Number of hidden neurons in our two fully connected (dense) layers.
# We increase the capacity to [128, 64] to handle the larger volume of flattened inputs (43,008).
DENSE_UNITS = [128, 64]

# Dropout probability to prevent overfitting in the dense layers.
DROPOUT_RATE = 0.3

class DeepNoiseCNN(nn.Module):
    """
    A lightweight 2D Convolutional Neural Network (CNN) in native PyTorch.
    Optimized for anomaly classification of Mel-spectrogram audio representations.
    
    In this version:
    - We removed Global Average Pooling (GAP) to preserve temporal and frequency locations.
    - We flatten the output of Conv Block 3 (128 channels * 16 height * 21 width = 43,008 features).
    - We added a deeper classification head: Linear(43008 -> 128) -> Linear(128 -> 64) -> Output(4).
    """
    def __init__(self, num_classes=4):
        super(DeepNoiseCNN, self).__init__()
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 1
        # ==========================================
        # Input shape: (Batch, 1, 128, 173)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_FILTERS[0], kernel_size=KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV_FILTERS[0])
        # Downsamples 128x173 to 64x86
        self.pool1 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 2
        # ==========================================
        # Input shape: (Batch, 32, 64, 86)
        self.conv2 = nn.Conv2d(in_channels=CONV_FILTERS[0], out_channels=CONV_FILTERS[1], kernel_size=KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV_FILTERS[1])
        # Downsamples 64x86 to 32x43
        self.pool2 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 3
        # ==========================================
        # Input shape: (Batch, 64, 32, 43)
        self.conv3 = nn.Conv2d(in_channels=CONV_FILTERS[1], out_channels=CONV_FILTERS[2], kernel_size=KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV_FILTERS[2])
        # Downsamples 32x43 to 16x21 (using floor division for odd dimensions)
        self.pool3 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # DEEPER CLASSIFICATION DENSE BLOCK
        # ==========================================
        
        # 1. First Dense Layer
        # - Input: 128 channels * 16 height * 21 width = 43,008 features.
        # - Output: 128 hidden units.
        self.fc1 = nn.Linear(CONV_FILTERS[2] * 16 * 21, DENSE_UNITS[0])
        
        # 2. Second Dense Layer
        # - Input: 128 units from fc1.
        # - Output: 64 hidden units.
        self.fc2 = nn.Linear(DENSE_UNITS[0], DENSE_UNITS[1])
        
        # 3. Output Classification Layer
        # - Maps the 64 hidden units to the 4 output logits (class scores).
        self.fc3 = nn.Linear(DENSE_UNITS[1], num_classes)
        
        # Regularization and Activations
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Defines the computation graph of the model.
        Input shape: (Batch, Height, Width, Channels) -> e.g., (16, 128, 173, 1)
        """
        # Step 1: Reshape input layout if it is channels-last to channels-first (N, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
            
        # Step 2: Feed through Conv Blocks
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Step 3: Flatten the 3D conv maps to 1D feature vectors
        # Converts shape from (Batch, 128, 16, 21) directly to (Batch, 43008), preserving all spatial coordinates.
        # We use .reshape() instead of .view() because x is non-contiguous after the permute/transpose step.
        x = x.reshape(x.size(0), -1)
        
        # Step 4: First Dense Layer with ReLU and Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        
        # Step 5: Second Dense Layer with ReLU and Dropout (Deeper classification head)
        x = self.dropout(self.relu(self.fc2(x)))
        
        # Step 6: Output logits
        # Yields shape (Batch, 4)
        x = self.fc3(x)
        return x

def build_cnn_model(num_classes=4):
    """
    Helper function to instantiate and return a DeepNoiseCNN model object.
    """
    return DeepNoiseCNN(num_classes=num_classes)

if __name__ == "__main__":
    # If run directly as a script, build and display the model structure and test input shape routing
    model = build_cnn_model()
    print("=== Model Architecture Structure ===")
    print(model)
    
    # Test feed-forward path shape with a random dummy spectrogram tensor
    dummy_input = torch.randn(16, 128, 173, 1)
    outputs = model(dummy_input)
    print("\n=== Dry-Run Shape Verification ===")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {outputs.shape} (Expected: [16, 4])")
