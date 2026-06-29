"""
Training Script for Image Classification Models
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from datetime import datetime
from typing import Tuple, Dict, List
import pickle

# Import models
import sys
sys.path.append(os.path.dirname(__file__))

from models_cnn_model import create_custom_cnn
from models_mobilenet_model import create_mobilenetv2
from models_resnet_model import create_resnet50
from utils_augmentation import augment_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage model training."""
    
    def __init__(self, model_name: str, model: keras.Model,
                 batch_size: int = 32,
                 epochs: int = 50,
                 validation_split: float = 0.2):
        """
        Initialize trainer.
        
        Args:
            model_name: Name of model
            model: Keras model
            batch_size: Batch size
            epochs: Number of epochs
            validation_split: Validation split ratio
        """
        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.history = None
        self.training_time = 0
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             callbacks: List = None) -> Dict:
        """
        Train model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            callbacks: Keras callbacks
        
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
        
        logger.info(f"Starting training for {self.model_name}")
        
        start_time = datetime.now()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val) if X_val is not None else self.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        self.training_time = (end_time - start_time).total_seconds()
        
        self.history = history.history
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def save_model(self, path: str) -> None:
        """
        Save model.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def save_history(self, path: str) -> None:
        """
        Save training history.
        
        Args:
            path: Path to save history
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {k: [float(v) for v in vals] 
                          for k, vals in self.history.items()}
            json.dump(history_json, f, indent=4)
        logger.info(f"History saved to {path}")
    
    def get_model_size(self) -> float:
        """
        Get model size in MB.
        
        Returns:
            Model size in MB
        """
        total_params = self.model.count_params()
        # Approximate size: 4 bytes per float32 parameter
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb


class DataLoader:
    """Load and prepare data."""
    
    @staticmethod
    def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CIFAR-10 dataset.
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        logger.info(f"CIFAR-10 loaded: {X_train.shape}, {X_test.shape}")
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def prepare_data(X: np.ndarray, y: np.ndarray,
                    num_classes: int = 10,
                    target_size: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            X: Images
            y: Labels
            num_classes: Number of classes
            target_size: Resize to this size
        
        Returns:
            Prepared images and one-hot encoded labels
        """
        if target_size:
            import cv2
            X_resized = np.array([cv2.resize(img, target_size) for img in X])
            X = X_resized
        
        # One-hot encode labels
        y = keras.utils.to_categorical(y, num_classes)
        
        return X, y


def train_all_models(dataset_name: str = 'cifar10',
                    epochs: int = 20,
                    batch_size: int = 32) -> Dict:
    """
    Train all three models.
    
    Args:
        dataset_name: Name of dataset
        epochs: Number of epochs
        batch_size: Batch size
    
    Returns:
        Dictionary with training results
    """
    # Load data
    logger.info("Loading data...")
    loader = DataLoader()
    X_train, y_train, X_test, y_test = loader.load_cifar10()
    
    # Prepare data
    num_classes = 10
    X_train, y_train = loader.prepare_data(X_train, y_train, num_classes)
    X_test, y_test = loader.prepare_data(X_test, y_test, num_classes)
    
    # Split validation
    split_idx = int(len(X_train) * 0.8)
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
    
    results = {}
    
    # Train Custom CNN
    logger.info("\n=== Training Custom CNN ===")
    cnn_model = create_custom_cnn(input_shape=(32, 32, 3), num_classes=num_classes)
    cnn_trainer = ModelTrainer('Custom_CNN', cnn_model, 
                              batch_size=batch_size, epochs=epochs)
    cnn_history = cnn_trainer.train(X_train_split, y_train_split, X_val, y_val)
    
    results['Custom_CNN'] = {
        'model': cnn_model,
        'trainer': cnn_trainer,
        'history': cnn_history,
        'training_time': cnn_trainer.training_time
    }
    
    # Train MobileNetV2
    logger.info("\n=== Training MobileNetV2 ===")
    X_train_mobile = keras.applications.mobilenet_v2.preprocess_input(
        X_train_split * 255
    )
    X_val_mobile = keras.applications.mobilenet_v2.preprocess_input(
        X_val * 255
    )
    X_test_mobile = keras.applications.mobilenet_v2.preprocess_input(
        X_test * 255
    )
    
    # Resize for MobileNetV2
    X_train_mobile = np.array([
        keras.preprocessing.image.smart_resize(img, (224, 224))
        for img in (X_train_split * 255).astype(np.uint8)
    ]) / 255.0
    X_val_mobile = np.array([
        keras.preprocessing.image.smart_resize(img, (224, 224))
        for img in (X_val * 255).astype(np.uint8)
    ]) / 255.0
    
    mobilenet_model = create_mobilenetv2(
        input_shape=(224, 224, 3), num_classes=num_classes
    )
    mobilenet_trainer = ModelTrainer('MobileNetV2', mobilenet_model,
                                    batch_size=batch_size, epochs=epochs)
    mobilenet_history = mobilenet_trainer.train(
        X_train_mobile, y_train_split, X_val_mobile, y_val
    )
    
    results['MobileNetV2'] = {
        'model': mobilenet_model,
        'trainer': mobilenet_trainer,
        'history': mobilenet_history,
        'training_time': mobilenet_trainer.training_time
    }
    
    # Train ResNet50
    logger.info("\n=== Training ResNet50 ===")
    X_train_resnet = np.array([
        keras.preprocessing.image.smart_resize(img, (224, 224))
        for img in (X_train_split * 255).astype(np.uint8)
    ]) / 255.0
    X_val_resnet = np.array([
        keras.preprocessing.image.smart_resize(img, (224, 224))
        for img in (X_val * 255).astype(np.uint8)
    ]) / 255.0
    
    resnet_model = create_resnet50(
        input_shape=(224, 224, 3), num_classes=num_classes
    )
    resnet_trainer = ModelTrainer('ResNet50', resnet_model,
                                 batch_size=batch_size, epochs=epochs)
    resnet_history = resnet_trainer.train(
        X_train_resnet, y_train_split, X_val_resnet, y_val
    )
    
    results['ResNet50'] = {
        'model': resnet_model,
        'trainer': resnet_trainer,
        'history': resnet_history,
        'training_time': resnet_trainer.training_time
    }
    
    # Evaluate models
    logger.info("\n=== Evaluating Models ===")
    for model_name, data in results.items():
        model = data['model']
        
        if 'Mobile' in model_name:
            eval_data = X_test_mobile
        elif 'ResNet' in model_name:
            eval_data = X_test_resnet
        else:
            eval_data = X_test
        
        test_loss, test_acc = model.evaluate(eval_data, y_test, verbose=0)
        logger.info(f"{model_name}: Test Accuracy = {test_acc:.4f}, Loss = {test_loss:.4f}")
        
        data['test_accuracy'] = float(test_acc)
        data['test_loss'] = float(test_loss)
    
    return results


if __name__ == "__main__":
    # Train all models
    results = train_all_models(epochs=20, batch_size=32)
    
    # Save models
    os.makedirs('saved_models', exist_ok=True)
    for model_name, data in results.items():
        path = f'saved_models/{model_name}_model.h5'
        data['trainer'].save_model(path)
        
        history_path = f'saved_models/{model_name}_history.json'
        data['trainer'].save_history(history_path)
    
    # Print summary
    logger.info("\n=== Training Summary ===")
    for model_name, data in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Training Time: {data['training_time']:.2f}s")
        logger.info(f"  Test Accuracy: {data['test_accuracy']:.4f}")
        logger.info(f"  Test Loss: {data['test_loss']:.4f}")
