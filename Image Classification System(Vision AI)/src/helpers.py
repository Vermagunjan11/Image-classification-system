"""
Helper Utilities and Common Functions for VisionAI
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pickle


# Setup logging
def setup_logging(log_file: str = None, level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Log file path
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('VisionAI')
    logger.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# File operations
class FileManager:
    """Manage file operations."""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Create directory if not exists."""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict, path: str) -> None:
        """Save dictionary to JSON file."""
        FileManager.ensure_directory(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    
    @staticmethod
    def load_json(path: str) -> Dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, path: str) -> None:
        """Save object to pickle file."""
        FileManager.ensure_directory(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(path: str) -> Any:
        """Load pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def get_file_size(path: str) -> float:
        """Get file size in MB."""
        return os.path.getsize(path) / (1024 * 1024)


# Time utilities
class TimeUtils:
    """Time-related utilities."""
    
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


# Array utilities
class ArrayUtils:
    """NumPy array utilities."""
    
    @staticmethod
    def normalize(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize array."""
        if method == 'minmax':
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)
        elif method == 'zscore':
            return (arr - arr.mean()) / (arr.std() + 1e-7)
        return arr
    
    @staticmethod
    def get_top_k_indices(arr: np.ndarray, k: int) -> np.ndarray:
        """Get indices of top k values."""
        return np.argsort(arr)[-k:][::-1]
    
    @staticmethod
    def batch_iterator(arr: np.ndarray,
                      batch_size: int):
        """Iterate over array in batches."""
        num_samples = len(arr)
        for i in range(0, num_samples, batch_size):
            yield arr[i:i+batch_size]


# String utilities
class StringUtils:
    """String manipulation utilities."""
    
    @staticmethod
    def truncate(text: str, max_length: int = 50) -> str:
        """Truncate string."""
        if len(text) > max_length:
            return text[:max_length-3] + '...'
        return text
    
    @staticmethod
    def format_number(num: float, decimals: int = 2) -> str:
        """Format number."""
        return f"{num:.{decimals}f}"
    
    @staticmethod
    def format_percentage(num: float, decimals: int = 2) -> str:
        """Format as percentage."""
        return f"{num*100:.{decimals}f}%"


# Validation utilities
class ValidationUtils:
    """Input validation utilities."""
    
    @staticmethod
    def validate_image_path(path: str) -> bool:
        """Validate image file path."""
        if not os.path.isfile(path):
            return False
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        return os.path.splitext(path)[1].lower() in valid_extensions
    
    @staticmethod
    def validate_model_path(path: str) -> bool:
        """Validate model file path."""
        if not os.path.isfile(path):
            return False
        
        return path.endswith('.h5') or path.endswith('.keras')
    
    @staticmethod
    def validate_data_shape(data: np.ndarray,
                          expected_shape: Tuple) -> bool:
        """Validate data shape."""
        if len(data.shape) != len(expected_shape):
            return False
        
        for actual, expected in zip(data.shape, expected_shape):
            if expected != -1 and actual != expected:
                return False
        
        return True


# Metrics utilities
class MetricsUtils:
    """Metrics calculation utilities."""
    
    @staticmethod
    def calculate_top_k_accuracy(y_true: np.ndarray,
                                y_pred_proba: np.ndarray,
                                k: int = 5) -> float:
        """Calculate top-k accuracy."""
        top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        correct = sum(1 for i, true_label in enumerate(y_true)
                     if true_label in top_k_pred[i])
        
        return correct / len(y_true)
    
    @staticmethod
    def calculate_confidence_stats(predictions: np.ndarray) -> Dict:
        """Calculate confidence statistics."""
        return {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max()),
            'median': float(np.median(predictions))
        }


# Report generation
class ReportGenerator:
    """Generate reports."""
    
    @staticmethod
    def generate_summary(metrics: Dict) -> str:
        """Generate text summary of metrics."""
        summary = []
        summary.append("=" * 50)
        summary.append("Model Evaluation Summary")
        summary.append("=" * 50)
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                summary.append(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        summary.append(f"  {k}: {v:.4f}")
                    else:
                        summary.append(f"  {k}: {v}")
            else:
                if isinstance(value, float):
                    summary.append(f"{key}: {value:.4f}")
                else:
                    summary.append(f"{key}: {value}")
        
        summary.append("=" * 50)
        return '\n'.join(summary)


# Cache management
class CacheManager:
    """Simple cache management."""
    
    def __init__(self, max_size: int = 100):
        """Initialize cache."""
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def set(self, key: str, value: Any) -> None:
        """Set cache value."""
        if key in self.cache:
            self.access_order.remove(key)
        
        self.cache[key] = value
        self.access_order.append(key)
        
        # Remove oldest if cache is full
        if len(self.cache) > self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
    
    def get(self, key: str) -> Any:
        """Get cache value."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


# Progress tracking
class ProgressTracker:
    """Track progress of operations."""
    
    def __init__(self, total: int, name: str = "Progress"):
        """Initialize tracker."""
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = datetime.now()
    
    def update(self, amount: int = 1) -> None:
        """Update progress."""
        self.current += amount
    
    def get_progress_percent(self) -> float:
        """Get progress percentage."""
        return (self.current / self.total) * 100
    
    def get_eta(self) -> str:
        """Estimate time to completion."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        return TimeUtils.format_duration(remaining)
    
    def __str__(self) -> str:
        """String representation."""
        percent = self.get_progress_percent()
        eta = self.get_eta()
        return f"{self.name}: {percent:.1f}% (ETA: {eta})"


# System utilities
class SystemUtils:
    """System-related utilities."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    @staticmethod
    def check_gpu_available() -> bool:
        """Check if GPU is available."""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except:
            return False
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get GPU information."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return {
                'available': len(gpus) > 0,
                'count': len(gpus),
                'devices': [str(gpu) for gpu in gpus]
            }
        except:
            return {'available': False, 'count': 0, 'devices': []}


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging()
    logger.info("VisionAI utilities loaded successfully!")
    
    # Create directories
    FileManager.ensure_directory('test_output')
    
    # Test time utilities
    print(f"Current timestamp: {TimeUtils.get_timestamp()}")
    print(f"Duration format: {TimeUtils.format_duration(3665)}")
    
    # Test string utilities
    print(f"Percentage format: {StringUtils.format_percentage(0.95)}")
    
    # Test system utilities
    print(f"GPU available: {SystemUtils.check_gpu_available()}")
