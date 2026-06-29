"""
Configuration Module for VisionAI
"""

import os
from typing import Dict, List, Tuple


class Config:
    """Base configuration."""
    
    # Project
    PROJECT_NAME = "VisionAI"
    VERSION = "1.0.0"
    DEBUG = False
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Models
    MODELS = {
        'custom_cnn': {
            'name': 'Custom CNN',
            'input_shape': (32, 32, 3),
            'pretrained': False,
            'model_type': 'custom'
        },
        'mobilenetv2': {
            'name': 'MobileNetV2',
            'input_shape': (224, 224, 3),
            'pretrained': True,
            'model_type': 'transfer'
        },
        'resnet50': {
            'name': 'ResNet50',
            'input_shape': (224, 224, 3),
            'pretrained': True,
            'model_type': 'transfer'
        }
    }
    
    # Dataset
    DATASET_NAME = 'cifar10'
    CIFAR10_CLASSES = [
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
        'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    ]
    NUM_CLASSES = 10
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5
    
    # Data Augmentation
    AUGMENTATION_PARAMS = {
        'rotation': True,
        'horizontal_flip': True,
        'vertical_flip': True,
        'shift': True,
        'zoom': True,
        'brightness': True,
        'contrast': True,
        'shear': True
    }
    
    # Preprocessing
    NORMALIZE_METHOD = 'minmax'  # 'minmax' or 'zscore'
    
    # Inference
    CONFIDENCE_THRESHOLD = 0.5
    TOP_K = 5
    
    # Image Upload
    MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
    
    # Performance
    USE_GPU = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # UI Settings
    THEME = 'dark'
    SHOW_METRICS = True
    SHOW_HEATMAP = True
    
    # Cache
    ENABLE_CACHE = True
    CACHE_SIZE_MB = 100


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    USE_GPU = True


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    BATCH_SIZE = 8
    EPOCHS = 2


# Select configuration based on environment
def get_config(env: str = 'development') -> Config:
    """
    Get configuration based on environment.
    
    Args:
        env: Environment name
    
    Returns:
        Configuration object
    """
    configs = {
        'development': DevelopmentConfig(),
        'production': ProductionConfig(),
        'testing': TestingConfig()
    }
    
    return configs.get(env, DevelopmentConfig())


# Default config
config = get_config()


# Model Configurations
MODEL_CONFIGS = {
    'custom_cnn': {
        'architecture': 'Custom CNN',
        'layers': 3,
        'input_shape': (32, 32, 3),
        'dropout': 0.5,
        'batch_norm': True,
        'description': 'Lightweight custom CNN for image classification'
    },
    'mobilenetv2': {
        'architecture': 'MobileNetV2',
        'layers': 53,
        'input_shape': (224, 224, 3),
        'dropout': 0.3,
        'batch_norm': True,
        'pretrained': True,
        'description': 'Efficient mobile-first architecture with transfer learning'
    },
    'resnet50': {
        'architecture': 'ResNet50',
        'layers': 50,
        'input_shape': (224, 224, 3),
        'dropout': 0.4,
        'batch_norm': True,
        'pretrained': True,
        'description': 'Deep residual network with transfer learning'
    }
}


# Training Hyperparameters
TRAINING_CONFIG = {
    'custom_cnn': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy'
    },
    'mobilenetv2': {
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'fine_tune_epochs': 10,
        'fine_tune_lr': 0.0001
    },
    'resnet50': {
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'fine_tune_epochs': 10,
        'fine_tune_lr': 0.00001
    }
}


# Performance Targets
PERFORMANCE_TARGETS = {
    'custom_cnn': {
        'accuracy': 0.82,
        'precision': 0.81,
        'recall': 0.82,
        'f1_score': 0.81
    },
    'mobilenetv2': {
        'accuracy': 0.91,
        'precision': 0.90,
        'recall': 0.91,
        'f1_score': 0.90
    },
    'resnet50': {
        'accuracy': 0.94,
        'precision': 0.93,
        'recall': 0.94,
        'f1_score': 0.93
    }
}


def create_directories():
    """Create necessary directories."""
    directories = [
        config.DATA_DIR,
        config.MODEL_DIR,
        config.OUTPUT_DIR,
        config.LOGS_DIR,
        os.path.join(config.OUTPUT_DIR, 'reports'),
        os.path.join(config.OUTPUT_DIR, 'predictions'),
        os.path.join(config.OUTPUT_DIR, 'heatmaps')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    # Test configuration
    print(f"Project: {config.PROJECT_NAME} v{config.VERSION}")
    print(f"Environment: development")
    print(f"Models: {list(config.MODELS.keys())}")
    print(f"Classes: {config.NUM_CLASSES}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    
    # Create directories
    create_directories()
    print("Directories created successfully!")
