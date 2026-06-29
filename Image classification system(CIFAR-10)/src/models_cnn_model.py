"""
Custom CNN Model for Image Classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List


class CustomCNN:
    """
    Custom Convolutional Neural Network for image classification.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3), 
                 num_classes: int = 10,
                 dropout_rate: float = 0.5):
        """
        Initialize CNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build(self) -> keras.Model:
        """
        Build the CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_model(self) -> keras.Model:
        """Get the compiled model."""
        if self.model is None:
            self.build()
        return self.model
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()


def create_custom_cnn(input_shape: Tuple[int, int, int] = (32, 32, 3),
                     num_classes: int = 10) -> keras.Model:
    """
    Factory function to create and return a custom CNN model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
    
    Returns:
        Compiled Keras model
    """
    cnn = CustomCNN(input_shape=input_shape, num_classes=num_classes)
    return cnn.build()
