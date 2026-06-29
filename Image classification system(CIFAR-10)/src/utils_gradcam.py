"""
Grad-CAM Implementation for Model Explainability
"""

import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    for neural network interpretability.
    """
    
    def __init__(self, model: tf.keras.Model, layer_name: str):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name: Name of convolutional layer to visualize
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = None
        self._build_grad_model()
    
    def _build_grad_model(self) -> None:
        """Build gradient model."""
        try:
            # Get the layer
            layer = self.model.get_layer(self.layer_name)
            
            # Create model that outputs both predictions and layer outputs
            self.grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [self.model.output, layer.output]
            )
        except Exception as e:
            logger.error(f"Error building grad model: {e}")
            raise
    
    def generate_heatmap(self, image: np.ndarray,
                        pred_index: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image (batch or single)
            pred_index: Index of class to visualize
        
        Returns:
            Heatmap array
        """
        # Add batch dimension if needed
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # Watch the image
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            predictions, conv_outputs = self.grad_model(image)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Calculate gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, image: np.ndarray,
                       heatmap: np.ndarray,
                       alpha: float = 0.5,
                       cmap: str = 'jet') -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            alpha: Blending factor
            cmap: Colormap name
        
        Returns:
            Image with overlaid heatmap
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, 
                                     (image.shape[1], image.shape[0]))
        
        # Normalize image if needed
        if image.max() <= 1:
            image_display = (image * 255).astype(np.uint8)
        else:
            image_display = image.astype(np.uint8)
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlaid = cv2.addWeighted(
            image_display, 1 - alpha,
            heatmap_colored, alpha,
            0
        )
        
        return overlaid


class LayerActivationVisualizer:
    """Visualize layer activations."""
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize visualizer.
        
        Args:
            model: Keras model
        """
        self.model = model
    
    def get_layer_outputs(self, image: np.ndarray,
                         layer_name: str) -> np.ndarray:
        """
        Get outputs of specific layer.
        
        Args:
            image: Input image
            layer_name: Layer name
        
        Returns:
            Layer outputs
        """
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        layer_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        outputs = layer_model(image)
        return outputs.numpy()
    
    def visualize_filters(self, layer_outputs: np.ndarray,
                         num_filters: int = 16) -> List[np.ndarray]:
        """
        Visualize filter activations.
        
        Args:
            layer_outputs: Layer output tensor
            num_filters: Number of filters to visualize
        
        Returns:
            List of filter visualizations
        """
        visualizations = []
        
        # Get number of channels
        num_channels = layer_outputs.shape[-1]
        
        for i in range(min(num_filters, num_channels)):
            # Extract filter activation
            filter_output = layer_outputs[0, :, :, i]
            
            # Normalize to [0, 1]
            filter_output = (filter_output - filter_output.min()) / \
                           (filter_output.max() - filter_output.min() + 1e-7)
            
            visualizations.append(filter_output.numpy() if isinstance(filter_output, tf.Tensor) else filter_output)
        
        return visualizations


def visualize_prediction_path(model: tf.keras.Model,
                             image: np.ndarray,
                             layer_names: List[str]) -> dict:
    """
    Visualize predictions through multiple layers.
    
    Args:
        model: Keras model
        image: Input image
        layer_names: List of layer names to visualize
    
    Returns:
        Dictionary of layer outputs
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    
    outputs = {}
    
    for layer_name in layer_names:
        try:
            layer = model.get_layer(layer_name)
            layer_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=layer.output
            )
            outputs[layer_name] = layer_model(image).numpy()
        except Exception as e:
            logger.warning(f"Could not get layer {layer_name}: {e}")
    
    return outputs


def create_feature_visualization(model: tf.keras.Model,
                                num_filters: int = 16,
                                layer_name: str = None) -> np.ndarray:
    """
    Create feature visualization by optimizing input.
    
    Args:
        model: Keras model
        num_filters: Number of filters
        layer_name: Layer to visualize
    
    Returns:
        Visualization array
    """
    # This is a simplified version
    # Full implementation would optimize input image
    
    input_shape = model.input_shape[1:]
    random_input = np.random.randn(1, *input_shape).astype(np.float32)
    
    return random_input[0]
