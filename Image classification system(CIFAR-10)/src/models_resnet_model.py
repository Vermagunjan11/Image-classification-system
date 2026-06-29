"""
ResNet50 Transfer Learning Model for Image Classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


class ResNet50Model:
    """
    ResNet50 model with transfer learning for image classification.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 10,
                 freeze_base: bool = True,
                 dropout_rate: float = 0.4):
        """
        Initialize ResNet50 model.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            freeze_base: Whether to freeze base model weights
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.freeze_base = freeze_base
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build(self) -> keras.Model:
        """
        Build ResNet50 model with custom top layers.
        
        Returns:
            Compiled Keras model
        """
        # Load pretrained ResNet50
        base_model = keras.applications.ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights
        if self.freeze_base:
            base_model.trainable = False
        
        # Create model
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Rescaling(1./255.0),  # Normalize
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
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
    
    def unfreeze_base(self, num_layers: int = 20) -> None:
        """
        Unfreeze top layers of base model for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from top
        """
        if self.model is None:
            self.build()
        
        base_model = self.model.layers[1]  # Get base model
        
        # Unfreeze top layers
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
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


def create_resnet50(input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 10) -> keras.Model:
    """
    Factory function to create ResNet50 model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
    
    Returns:
        Compiled Keras model
    """
    resnet = ResNet50Model(input_shape=input_shape, num_classes=num_classes)
    return resnet.build()
