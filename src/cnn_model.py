import os

# Ensure Keras runs on the PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models

# --- CNN Architecture Hyperparameters ---
# Convolutional filters for blocks 1, 2, and 3
CONV_FILTERS = [32, 64, 128]

# Convolutional kernel size (3x3 is standard for extracting spatial audio patterns)
KERNEL_SIZE = (3, 3)

# Pooling size (2x2 reduces spatial dimensions by 50%)
POOL_SIZE = (2, 2)

# Dense layer size before output classification
DENSE_UNITS = 64

# Dropout rate to prevent nodes from co-adapting (prevents overfitting)
DROPOUT_RATE = 0.3


def build_cnn_model(input_shape=(128, 173, 1), num_classes=4):
    """
    Constructs a lightweight 2D Convolutional Neural Network (CNN)
    suitable for classification of Mel-spectrogram images.
    
    Structure:
    - Input Layer: (128, 173, 1) -> 128 frequency bins, 173 time frames, 1 channel (greyscale)
    - Conv Block 1: Conv2D (32 filters) -> BatchNormalization -> ReLU -> MaxPooling2D
    - Conv Block 2: Conv2D (64 filters) -> BatchNormalization -> ReLU -> MaxPooling2D
    - Conv Block 3: Conv2D (128 filters) -> BatchNormalization -> ReLU -> MaxPooling2D
    - Pooling: GlobalAveragePooling2D (reduces dimensions without exploding fully-connected parameters)
    - Dense Block: Dense (64 units) -> ReLU -> Dropout (0.3)
    - Output Layer: Dense (4 units) -> Softmax (probabilities for the 4 classes)
    """
    # 1. Input Layer
    inputs = keras.Input(shape=input_shape, name="mel_spectrogram_input")
    x = inputs
    
    # 2. Convolutional Blocks
    for i, filters in enumerate(CONV_FILTERS):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=KERNEL_SIZE,
            padding="same",
            name=f"conv_block_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"batch_norm_{i+1}")(x)
        x = layers.ReLU(name=f"relu_{i+1}")(x)
        x = layers.MaxPooling2D(pool_size=POOL_SIZE, name=f"max_pool_{i+1}")(x)
        
    # 3. Dimensionality Reduction
    # GlobalAveragePooling2D calculates the average of each feature map,
    # reducing (Height, Width, Channels) to (Channels,).
    # This prevents the explosive increase in parameters caused by standard flattening.
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    
    # 4. Fully Connected (Dense) Block
    x = layers.Dense(units=DENSE_UNITS, activation="relu", name="dense_dense")(x)
    x = layers.Dropout(rate=DROPOUT_RATE, name="dropout")(x)
    
    # 5. Output Layer (Softmax activation outputs probability distribution over target classes)
    outputs = layers.Dense(units=num_classes, activation="softmax", name="output_classification")(x)
    
    # Instantiate Model
    model = models.Model(inputs=inputs, outputs=outputs, name="DeepNoise_Lightweight_CNN")
    
    return model


if __name__ == "__main__":
    # Test compilation and print model summary
    model = build_cnn_model()
    model.summary()
