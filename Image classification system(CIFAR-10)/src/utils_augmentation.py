"""
Data Augmentation Utilities for Image Enhancement
"""

import cv2
import numpy as np
from typing import Tuple, Callable, List
import logging

logger = logging.getLogger(__name__)


class DataAugmentor:
    """Apply various data augmentation techniques."""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
        
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated
    
    @staticmethod
    def horizontal_flip(image: np.ndarray) -> np.ndarray:
        """
        Flip image horizontally.
        
        Args:
            image: Input image
        
        Returns:
            Flipped image
        """
        return cv2.flip(image, 1)
    
    @staticmethod
    def vertical_flip(image: np.ndarray) -> np.ndarray:
        """
        Flip image vertically.
        
        Args:
            image: Input image
        
        Returns:
            Flipped image
        """
        return cv2.flip(image, 0)
    
    @staticmethod
    def shift(image: np.ndarray, 
             shift_x: float = 0.0,
             shift_y: float = 0.0) -> np.ndarray:
        """
        Shift image horizontally and vertically.
        
        Args:
            image: Input image
            shift_x: Horizontal shift ratio (-0.5 to 0.5)
            shift_y: Vertical shift ratio (-0.5 to 0.5)
        
        Returns:
            Shifted image
        """
        h, w = image.shape[:2]
        shift_x_pixels = int(w * shift_x)
        shift_y_pixels = int(h * shift_y)
        
        translation_matrix = np.float32([
            [1, 0, shift_x_pixels],
            [0, 1, shift_y_pixels]
        ])
        
        shifted = cv2.warpAffine(image, translation_matrix, (w, h))
        return shifted
    
    @staticmethod
    def zoom(image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """
        Zoom image.
        
        Args:
            image: Input image
            zoom_factor: Zoom factor (>1 for zoom in, <1 for zoom out)
        
        Returns:
            Zoomed image
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        zoomed = cv2.resize(image, (new_w, new_h))
        
        if zoom_factor > 1:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            zoomed = zoomed[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad with zeros
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            zoomed = cv2.copyMakeBorder(zoomed, pad_h, pad_h, pad_w, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
        
        return zoomed
    
    @staticmethod
    def brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (>1 brighter, <1 darker)
        
        Returns:
            Brightness-adjusted image
        """
        image = image.astype(np.float32)
        image = image * factor
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    
    @staticmethod
    def contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (>1 more contrast, <1 less contrast)
        
        Returns:
            Contrast-adjusted image
        """
        image = image.astype(np.float32)
        mean = image.mean()
        image = (image - mean) * factor + mean
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    
    @staticmethod
    def shear(image: np.ndarray, shear_factor: float) -> np.ndarray:
        """
        Apply shear transformation.
        
        Args:
            image: Input image
            shear_factor: Shear factor
        
        Returns:
            Sheared image
        """
        h, w = image.shape[:2]
        shear_matrix = np.float32([
            [1, shear_factor, 0],
            [0, 1, 0]
        ])
        
        sheared = cv2.warpAffine(image, shear_matrix, (w, h))
        return sheared
    
    @staticmethod
    def elastic_deformation(image: np.ndarray,
                           alpha: float = 30,
                           sigma: float = 5) -> np.ndarray:
        """
        Apply elastic deformation.
        
        Args:
            image: Input image
            alpha: Deformation intensity
            sigma: Smoothness
        
        Returns:
            Deformed image
        """
        h, w = image.shape[:2]
        
        # Generate random displacement maps
        dx = np.random.randn(h, w) * sigma
        dy = np.random.randn(h, w) * sigma
        
        # Apply Gaussian blur to smooth
        dx = cv2.GaussianBlur(dx, (5, 5), 0) * alpha
        dy = cv2.GaussianBlur(dy, (5, 5), 0) * alpha
        
        # Create coordinate maps
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_map = (x + dx).astype(np.float32)
        y_map = (y + dy).astype(np.float32)
        
        # Apply remap
        deformed = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR)
        return deformed
    
    @staticmethod
    def noise(image: np.ndarray, noise_type: str = 'gaussian',
             amount: float = 0.01) -> np.ndarray:
        """
        Add noise to image.
        
        Args:
            image: Input image
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson')
            amount: Amount of noise
        
        Returns:
            Noisy image
        """
        image = image.astype(np.float32)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, amount * 255, image.shape)
            image = image + noise
        
        elif noise_type == 'salt_pepper':
            num_pixels = int(amount * image.size)
            for _ in range(num_pixels):
                y = np.random.randint(0, image.shape[0])
                x = np.random.randint(0, image.shape[1])
                image[y, x] = 0 if np.random.random() < 0.5 else 255
        
        elif noise_type == 'poisson':
            image = np.random.poisson(image / 255.0 * amount) * (255.0 / amount)
        
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)


def augment_image(image: np.ndarray,
                 augmentation_params: dict) -> np.ndarray:
    """
    Apply multiple augmentations to image.
    
    Args:
        image: Input image
        augmentation_params: Dictionary of augmentation parameters
    
    Returns:
        Augmented image
    """
    augmentor = DataAugmentor()
    
    if augmentation_params.get('rotation'):
        angle = np.random.uniform(-20, 20)
        image = augmentor.rotate(image, angle)
    
    if augmentation_params.get('horizontal_flip') and np.random.random() > 0.5:
        image = augmentor.horizontal_flip(image)
    
    if augmentation_params.get('vertical_flip') and np.random.random() > 0.5:
        image = augmentor.vertical_flip(image)
    
    if augmentation_params.get('shift'):
        shift_x = np.random.uniform(-0.1, 0.1)
        shift_y = np.random.uniform(-0.1, 0.1)
        image = augmentor.shift(image, shift_x, shift_y)
    
    if augmentation_params.get('zoom'):
        zoom_factor = np.random.uniform(0.9, 1.1)
        image = augmentor.zoom(image, zoom_factor)
    
    if augmentation_params.get('brightness'):
        factor = np.random.uniform(0.8, 1.2)
        image = augmentor.brightness(image, factor)
    
    if augmentation_params.get('contrast'):
        factor = np.random.uniform(0.8, 1.2)
        image = augmentor.contrast(image, factor)
    
    if augmentation_params.get('shear'):
        shear_factor = np.random.uniform(-0.2, 0.2)
        image = augmentor.shear(image, shear_factor)
    
    return image


def create_augmentation_samples(image: np.ndarray,
                               num_samples: int = 6) -> List[np.ndarray]:
    """
    Create multiple augmented samples of an image.
    
    Args:
        image: Input image
        num_samples: Number of samples to create
    
    Returns:
        List of augmented images
    """
    augmentation_params = {
        'rotation': True,
        'horizontal_flip': True,
        'vertical_flip': True,
        'shift': True,
        'zoom': True,
        'brightness': True,
        'contrast': True,
        'shear': True
    }
    
    samples = [image]
    for _ in range(num_samples - 1):
        augmented = augment_image(image.copy(), augmentation_params)
        samples.append(augmented)
    
    return samples
