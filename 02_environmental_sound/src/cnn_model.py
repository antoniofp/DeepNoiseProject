import torch
import torch.nn as nn

# --- CNN Architecture Hyperparameters ---
# The number of convolutional filters (feature maps) extracted in each block.
CONV_FILTERS = [32, 64, 128]

# The height and width of the 2D sliding filter window.
KERNEL_SIZE = 3

# The size of the Max Pooling window.
POOL_SIZE = 2

# Number of hidden units in the two fully connected (dense) layers.
# We map from 2,048 pooled features to 128, then 64, and finally to 8 output classes.
DENSE_UNITS = [128, 64]

# Dropout probability to prevent overfitting in the dense layers.
DROPOUT_RATE = 0.3

class DeepNoiseCNN(nn.Module):
    """
    A lightweight 2D Convolutional Neural Network (CNN) in native PyTorch.
    Optimized for acoustic event classification of Mel-spectrogram representations.
    
    This architecture utilizes an Adaptive Average Pooling layer to downsample feature maps
    to a fixed 4x4 grid before flattening. This design ensures that the fully connected layers
    receive a consistent 2,048 features, regardless of slight differences in input audio duration
    or spectrogram temporal dimensions, while maintaining spatial and temporal context.
    """
    def __init__(self, num_classes=8):
        super(DeepNoiseCNN, self).__init__()
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 1
        # ==========================================
        # Input shape: (Batch, 1, 128, 216)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_FILTERS[0], kernel_size=KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV_FILTERS[0])
        # Downsamples 128x216 to 64x108
        self.pool1 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 2
        # ==========================================
        # Input shape: (Batch, 32, 64, 108)
        self.conv2 = nn.Conv2d(in_channels=CONV_FILTERS[0], out_channels=CONV_FILTERS[1], kernel_size=KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV_FILTERS[1])
        # Downsamples 64x108 to 32x54
        self.pool2 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 3
        # ==========================================
        # Input shape: (Batch, 64, 32, 54)
        self.conv3 = nn.Conv2d(in_channels=CONV_FILTERS[1], out_channels=CONV_FILTERS[2], kernel_size=KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV_FILTERS[2])
        # Downsamples 32x54 to 16x27
        self.pool3 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # HYBRID ADAPTIVE POOLING LAYER
        # ==========================================
        # Instead of global average pooling to a 1x1 size, we pool to a 4x4 spatial grid.
        # This keeps critical relative time-frequency distributions intact.
        self.pool_adaptive = nn.AdaptiveAvgPool2d((4, 4))
        
        # ==========================================
        # CLASSIFICATION DENSE BLOCK
        # ==========================================
        # 1. First Dense Layer
        # - Input: 128 channels * 4 height * 4 width = 2,048 features.
        # - Output: 128 hidden units.
        self.fc1 = nn.Linear(CONV_FILTERS[2] * 4 * 4, DENSE_UNITS[0])
        
        # 2. Second Dense Layer
        # - Input: 128 units.
        # - Output: 64 hidden units.
        self.fc2 = nn.Linear(DENSE_UNITS[0], DENSE_UNITS[1])
        
        # 3. Output Classification Layer
        # - Maps the 64 hidden units to the 8 output logits (class scores).
        self.fc3 = nn.Linear(DENSE_UNITS[1], num_classes)
        
        # Regularization and Activations
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Defines the computation graph of the model.
        Input shape: (Batch, Height, Width, Channels) or (Batch, Channels, Height, Width)
        """
        # Step 1: Reshape input layout if it is channels-last to channels-first (N, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
            
        # Step 2: Feed through Conv Blocks
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Step 3: Run Adaptive Pooling to 4x4 grid
        # Output shape: (Batch, 128, 4, 4)
        x = self.pool_adaptive(x)
        
        # Step 4: Flatten the 4x4 maps to 1D feature vectors
        # Converts shape from (Batch, 128, 4, 4) directly to (Batch, 2048).
        x = x.reshape(x.size(0), -1)
        
        # Step 5: First Dense Layer with ReLU and Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        
        # Step 6: Second Dense Layer with ReLU and Dropout
        x = self.dropout(self.relu(self.fc2(x)))
        
        # Step 7: Output logits
        x = self.fc3(x)
        return x

def build_cnn_model(num_classes=8):
    """
    Helper function to instantiate and return a DeepNoiseCNN model object.
    """
    return DeepNoiseCNN(num_classes=num_classes)

if __name__ == "__main__":
    # Test model instantiation and input/output dimensions
    model = build_cnn_model(num_classes=8)
    print("=== Model Architecture Structure ===")
    print(model)
    
    # Verify shape flow with standard 5-second Mel-spectrogram input
    # Shape: (Batch=16, Height=128, Width=216, Channels=1)
    dummy_input = torch.randn(16, 128, 216, 1)
    outputs = model(dummy_input)
    print("\n=== Model Shape Verification ===")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {outputs.shape} (Expected: [16, 8])")
