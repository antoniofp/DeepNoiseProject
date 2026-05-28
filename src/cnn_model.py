import torch
import torch.nn as nn

# --- CNN Architecture Hyperparameters ---
# The number of convolutional filters (feature maps) extracted in each respective block.
# We start with 32 filters for low-level features, then increase to 64 and 128 to capture complex sound combinations.
CONV_FILTERS = [32, 64, 128]

# The height and width of the 2D sliding filter window.
# A 3x3 kernel size is a standard choice that captures local patterns without excessive parameters.
KERNEL_SIZE = 3

# The size of the Max Pooling window.
# A 2x2 pooling size halves the height and width dimensions, reducing spatial resolution by a factor of 4.
POOL_SIZE = 2

# Number of hidden neurons in the fully connected (dense) classification hidden layer.
DENSE_UNITS = 64

# The probability that any given hidden neuron will be randomly deactivated (zeroed) during training.
# 0.3 means 30% of the neurons are disabled on each step, forcing robust feature redundancy.
DROPOUT_RATE = 0.3

class DeepNoiseCNN(nn.Module):
    """
    A lightweight 2D Convolutional Neural Network (CNN) in native PyTorch.
    Designed for classification of Mel-spectrogram audio representations.
    
    This class inherits from PyTorch's base `nn.Module`. 
    In PyTorch, all custom neural network architectures must inherit from `nn.Module`
    so that PyTorch can automatically register layers, track parameters, and calculate gradients.
    """
    def __init__(self, num_classes=4):
        # We must call the parent class constructor (nn.Module) first.
        # This initializes the underlying PyTorch graph registration and layer tracking.
        super(DeepNoiseCNN, self).__init__()
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 1
        # ==========================================
        
        # 1. 2D Convolution Layer
        # - in_channels=1: Grayscale Mel-spectrogram inputs have a single audio channel.
        # - out_channels=32: The layer will learn 32 distinct 3x3 feature detector filters.
        # - padding=1: Adds a 1-pixel border of zeros to keep height & width constant after the convolution.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_FILTERS[0], kernel_size=KERNEL_SIZE, padding=1)
        
        # 2. 2D Batch Normalization
        # - normalizes the activations of the 32 feature channels across the batch.
        # - Prevents gradients from vanishing or exploding, allowing faster training.
        self.bn1 = nn.BatchNorm2d(CONV_FILTERS[0])
        
        # 3. 2D Max Pooling
        # - Slides a 2x2 window across the image and outputs only the maximum value in each window.
        # - Reduces height and width by half (downsampling), making the model less sensitive to the exact location of a sound.
        self.pool1 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        

        # CONVOLUTIONAL BLOCK 2        
        # Takes the 32 feature maps from Block 1 and produces 64 deeper feature maps.
        self.conv2 = nn.Conv2d(in_channels=CONV_FILTERS[0], out_channels=CONV_FILTERS[1], kernel_size=KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV_FILTERS[1])
        self.pool2 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # CONVOLUTIONAL BLOCK 3
        # ==========================================
        
        # Takes the 64 feature maps from Block 2 and produces 128 deep feature maps.
        self.conv3 = nn.Conv2d(in_channels=CONV_FILTERS[1], out_channels=CONV_FILTERS[2], kernel_size=KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV_FILTERS[2])
        self.pool3 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        
        # ==========================================
        # GLOBAL POOLING AND CLASSIFICATION LAYERS
        # ==========================================
        
        # 1. Global Average Pooling (GAP)
        # - Computes the average value of each of the 128 channels across its entire height and width.
        # - Reduces any spatial shape (e.g., 16x21) down to a simple 1x1 size.
        # - Greatly reduces parameters, makes the model input-size flexible, and fights overfitting.
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 2. Dense (Linear) Hidden Layer
        # - Maps the 128 averaged channel features down to 64 intermediate hidden units.
        self.fc1 = nn.Linear(CONV_FILTERS[2], DENSE_UNITS)
        
        # 3. Dropout Regularization
        # - Randomly zeroes out 30% of activations in the dense layer during training to prevent memorization.
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        
        # 4. Dense (Linear) Output Layer
        # - Maps the 64 hidden units to the final 4 output scores (one for each machine category).
        # - Outputs raw "logits" (unnormalized scores) compatible with PyTorch's CrossEntropyLoss.
        self.fc2 = nn.Linear(DENSE_UNITS, num_classes)
        
        # 5. Non-linear Activation Function
        # - Rectified Linear Unit: returns 0 if input is negative, and returns the input value if positive.
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Defines the computation graph of the model. 
        It describes how the input tensor `x` travels step-by-step through the layers.
        
        Input shape: (Batch Size, Height, Width, Channels) -> e.g., (16, 128, 173, 1)
        """
        # Step 1: Reshape input layout if it is channels-last
        # PyTorch requires (Batch, Channels, Height, Width).
        # permute(0, 3, 1, 2) swaps axes: Batch (0) stays first, Channel (3) goes second, Height (1) third, Width (2) fourth.
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
            
        # Step 2: Feed through Conv Block 1
        # Order: Convolution -> Batch Normalization -> ReLU Activation -> Max Pooling
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # Step 3: Feed through Conv Block 2
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # Step 4: Feed through Conv Block 3
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Step 5: Perform Global Average Pooling
        # Converts shape from (Batch, 128, Height, Width) to (Batch, 128, 1, 1)
        x = self.gap(x)
        
        # Step 6: Flatten spatial dimensions
        # view(Batch, -1) reshapes the tensor from (Batch, 128, 1, 1) to a flat (Batch, 128) per sample.
        x = x.view(x.size(0), -1)
        
        # Step 7: Feed through Hidden Dense Layer with Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        
        # Step 8: Calculate final output logits
        # Returns shape (Batch, 4) containing raw predictions for each machine class.
        x = self.fc2(x)
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
