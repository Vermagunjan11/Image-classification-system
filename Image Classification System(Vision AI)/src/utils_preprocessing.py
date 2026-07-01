"""
Image Preprocessing Utilities
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union, List
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handle image preprocessing operations."""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    @staticmethod
    def load_from_pil(pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array.
        
        Args:
            pil_image: PIL Image object
        
        Returns:
            Image as numpy array
        """
        return np.array(pil_image)
    
    @staticmethod
    def resize(image: np.ndarray, 
               target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (height, width)
        
        Returns:
            Resized image
        """
        try:
            resized = cv2.resize(image, (target_size[1], target_size[0]))
            return resized
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise
    
    @staticmethod
    def normalize(image: np.ndarray, 
                 method: str = 'minmax') -> np.ndarray:
        """
        Normalize image.
        
        Args:
            image: Input image
            method: Normalization method ('minmax' or 'zscore')
        
        Returns:
            Normalized image
        """
        if method == 'minmax':
            # Min-Max normalization to [0, 1]
            image = image.astype(np.float32) / 255.0
        elif method == 'zscore':
            # Z-score normalization
            image = image.astype(np.float32)
            image = (image - image.mean()) / (image.std() + 1e-7)
        
        return image
    
    @staticmethod
    def standardize(image: np.ndarray,
                   mean: List[float] = None,
                   std: List[float] = None) -> np.ndarray:
        """
        Standardize image using mean and std.
        
        Args:
            image: Input image
            mean: Mean values for each channel
            std: Standard deviation for each channel
        
        Returns:
            Standardized image
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        return image
    
    @staticmethod
    def preprocess_cnn(image: np.ndarray,
                      target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
        """
        Preprocess image for custom CNN.
        
        Args:
            image: Input image
            target_size: Target size
        
        Returns:
            Preprocessed image
        """
        image = ImagePreprocessor.resize(image, target_size)
        image = ImagePreprocessor.normalize(image, method='minmax')
        return image
    
    @staticmethod
    def preprocess_transfer(image: np.ndarray,
                           target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for transfer learning models.
        
        Args:
            image: Input image
            target_size: Target size
        
        Returns:
            Preprocessed image
        """
        image = ImagePreprocessor.resize(image, target_size)
        # Keep in [0, 255] range as transfer models contain Rescaling layers
        image = image.astype(np.float32)
        return image
    
    @staticmethod
    def add_batch_dimension(image: np.ndarray) -> np.ndarray:
        """
        Add batch dimension to image.
        
        Args:
            image: Input image
        
        Returns:
            Image with batch dimension
        """
        return np.expand_dims(image, axis=0)
    
    @staticmethod
    def remove_batch_dimension(image: np.ndarray) -> np.ndarray:
        """
        Remove batch dimension from image.
        
        Args:
            image: Input image with batch dimension
        
        Returns:
            Image without batch dimension
        """
        if image.ndim == 4:
            return image[0]
        return image


class DataNormalizer:
    """Normalize datasets."""
    
    @staticmethod
    def normalize_dataset(X: np.ndarray, 
                         method: str = 'minmax') -> Tuple[np.ndarray, dict]:
        """
        Normalize dataset and return normalization parameters.
        
        Args:
            X: Input dataset
            method: Normalization method
        
        Returns:
            Normalized dataset and normalization parameters
        """
        norm_params = {}
        
        if method == 'minmax':
            min_val = X.min()
            max_val = X.max()
            X_normalized = (X - min_val) / (max_val - min_val + 1e-7)
            norm_params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        elif method == 'zscore':
            mean_val = X.mean()
            std_val = X.std()
            X_normalized = (X - mean_val) / (std_val + 1e-7)
            norm_params = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
        
        return X_normalized, norm_params
    
    @staticmethod
    def denormalize(X: np.ndarray, norm_params: dict) -> np.ndarray:
        """
        Denormalize dataset.
        
        Args:
            X: Normalized dataset
            norm_params: Normalization parameters
        
        Returns:
            Denormalized dataset
        """
        method = norm_params.get('method', 'minmax')
        
        if method == 'minmax':
            min_val = norm_params['min']
            max_val = norm_params['max']
            return X * (max_val - min_val) + min_val
        
        elif method == 'zscore':
            mean_val = norm_params['mean']
            std_val = norm_params['std']
            return X * std_val + mean_val
        
        return X


def preprocess_batch(images: List[np.ndarray],
                    target_size: Tuple[int, int],
                    normalize: bool = True) -> np.ndarray:
    """
    Preprocess batch of images.
    
    Args:
        images: List of images
        target_size: Target size
        normalize: Whether to normalize
    
    Returns:
        Batch of preprocessed images
    """
    processed = []
    for img in images:
        if isinstance(img, Image.Image):
            img = ImagePreprocessor.load_from_pil(img)
        
        img = ImagePreprocessor.resize(img, target_size)
        
        if normalize:
            img = ImagePreprocessor.normalize(img)
        
        processed.append(img)
    
    return np.array(processed)
