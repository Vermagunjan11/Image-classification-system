"""
Prediction Script for Image Classification
"""

import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, List, Union
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ImageClassifier:
    """Perform image classification predictions."""
    
    def __init__(self, model_path: str,
                 class_names: List[str] = None,
                 input_shape: Tuple[int, int, int] = (32, 32, 3),
                 model_type: str = 'custom'):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to saved model
            class_names: List of class names
            input_shape: Input shape for model
            model_type: Type of model ('custom', 'mobilenet', 'resnet')
        """
        self.model_path = model_path
        self.class_names = class_names or self._default_classes()
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.load_model()
    
    def _default_classes(self) -> List[str]:
        """Get default CIFAR-10 classes."""
        return ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    def load_model(self) -> None:
        """Load trained model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image: Image path, numpy array, or PIL Image
        
        Returns:
            Preprocessed image
        """
        # Load image if path
        if isinstance(image, str):
            if image.startswith('http'):
                from PIL import Image as PILImage
                import requests
                response = requests.get(image)
                image = PILImage.open(io.BytesIO(response.content))
            else:
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize
        if self.model_type == 'custom':
            image = image.astype(np.float32) / 255.0
        else:  # Transfer learning models (built-in Rescaling layer handles normalization)
            image = image.astype(np.float32)
        
        return image
    
    def predict(self, image: Union[str, np.ndarray, Image.Image],
               return_top_k: int = 5) -> Dict:
        """
        Make prediction on image.
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_top_k: Return top K predictions
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        processed = self.preprocess_image(image)
        processed = np.expand_dims(processed, axis=0)
        
        # Make prediction
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top K
        top_k_indices = np.argsort(predictions)[-return_top_k:][::-1]
        
        result = {
            'predicted_class': self.class_names[np.argmax(predictions)],
            'confidence': float(np.max(predictions)),
            'all_predictions': {self.class_names[i]: float(predictions[i])
                              for i in range(len(predictions))},
            'top_k_predictions': [
                {'class': self.class_names[i],
                 'confidence': float(predictions[i])}
                for i in top_k_indices
            ]
        }
        
        logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
        
        return result
    
    def predict_batch(self, images: List[Union[str, np.ndarray]],
                     return_top_k: int = 5) -> List[Dict]:
        """
        Make predictions on batch of images.
        
        Args:
            images: List of images
            return_top_k: Return top K predictions
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, return_top_k)
            results.append(result)
        
        return results
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        summary_str = []
        self.model.summary(print_fn=summary_str.append)
        return '\n'.join(summary_str)


class PredictionAnalyzer:
    """Analyze predictions."""
    
    @staticmethod
    def analyze_confidence(predictions: Dict) -> Dict:
        """
        Analyze confidence of predictions.
        
        Args:
            predictions: Prediction dictionary
        
        Returns:
            Confidence analysis
        """
        conf = predictions['confidence']
        top_5 = predictions['top_k_predictions']
        
        # Calculate confidence gap
        if len(top_5) > 1:
            conf_gap = top_5[0]['confidence'] - top_5[1]['confidence']
        else:
            conf_gap = 1.0
        
        analysis = {
            'high_confidence': conf > 0.9,
            'medium_confidence': 0.7 <= conf <= 0.9,
            'low_confidence': conf < 0.7,
            'confidence_gap': conf_gap,
            'is_decisive': conf_gap > 0.2
        }
        
        return analysis
    
    @staticmethod
    def compare_predictions(predictions_list: List[Dict]) -> Dict:
        """
        Compare predictions across models.
        
        Args:
            predictions_list: List of predictions from different models
        
        Returns:
            Comparison analysis
        """
        predictions = [p['predicted_class'] for p in predictions_list]
        confidences = [p['confidence'] for p in predictions_list]
        
        comparison = {
            'agreement': len(set(predictions)) == 1,
            'unanimous_prediction': predictions[0] if len(set(predictions)) == 1 else None,
            'average_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'confidence_std': np.std(confidences)
        }
        
        return comparison


class ResultExporter:
    """Export predictions and results."""
    
    @staticmethod
    def export_to_json(predictions: Dict, output_path: str) -> None:
        """
        Export predictions to JSON.
        
        Args:
            predictions: Prediction dictionary
            output_path: Output file path
        """
        import json
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"Results exported to {output_path}")
    
    @staticmethod
    def export_to_csv(predictions_list: List[Dict],
                     output_path: str,
                     image_names: List[str] = None) -> None:
        """
        Export predictions to CSV.
        
        Args:
            predictions_list: List of prediction dictionaries
            output_path: Output file path
            image_names: List of image names
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            if not predictions_list:
                return
            
            fieldnames = ['image', 'predicted_class', 'confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for idx, pred in enumerate(predictions_list):
                row = {
                    'image': image_names[idx] if image_names else f'image_{idx}',
                    'predicted_class': pred['predicted_class'],
                    'confidence': f"{pred['confidence']:.4f}"
                }
                writer.writerow(row)
        
        logger.info(f"Results exported to {output_path}")


def create_prediction_cache():
    """Create a simple cache for predictions."""
    return {
        'images': {},
        'predictions': {}
    }


def predict_with_cache(classifier: ImageClassifier,
                      image_path: str,
                      cache: dict) -> Dict:
    """
    Make prediction with caching.
    
    Args:
        classifier: ImageClassifier instance
        image_path: Path to image
        cache: Cache dictionary
    
    Returns:
        Prediction dictionary
    """
    if image_path not in cache['predictions']:
        cache['predictions'][image_path] = classifier.predict(image_path)
    
    return cache['predictions'][image_path]


if __name__ == "__main__":
    # Example usage
    # Load model
    # classifier = ImageClassifier(
    #     model_path='saved_models/Custom_CNN_model.h5',
    #     input_shape=(32, 32, 3),
    #     model_type='custom'
    # )
    
    # # Make prediction
    # result = classifier.predict('path/to/image.jpg')
    # print(result)
    
    pass
