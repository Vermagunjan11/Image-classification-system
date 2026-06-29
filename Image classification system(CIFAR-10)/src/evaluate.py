"""
Model Evaluation Script
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self, model: tf.keras.Model,
                 class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
    
    def evaluate(self, X_test: np.ndarray,
                y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test images
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert one-hot to labels if needed
        if y_test.ndim > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_pred)
        precision = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred)
        
        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(
            y_test_labels, y_pred, cm
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return metrics
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    cm: np.ndarray) -> Dict:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cm: Confusion matrix
        
        Returns:
            Per-class metrics
        """
        num_classes = cm.shape[0]
        per_class = {}
        
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) \
                 if (precision + recall) > 0 else 0
            
            class_name = self.class_names[i] if self.class_names else f"class_{i}"
            
            per_class[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(cm[i, :].sum())
            }
        
        return per_class
    
    def get_classification_report(self, X_test: np.ndarray,
                                 y_test: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            X_test: Test images
            y_test: Test labels
        
        Returns:
            Classification report string
        """
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        if y_test.ndim > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
        
        report = classification_report(
            y_test_labels, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        return report
    
    def calculate_top_k_accuracy(self, X_test: np.ndarray,
                                y_test: np.ndarray,
                                k: int = 5) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            X_test: Test images
            y_test: Test labels
            k: Top k
        
        Returns:
            Top-k accuracy
        """
        y_pred_proba = self.model.predict(X_test, verbose=0)
        
        if y_test.ndim > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
        
        # Get top k predictions for each sample
        top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        # Check if true label is in top k
        correct = sum(1 for i, true_label in enumerate(y_test_labels)
                     if true_label in top_k_pred[i])
        
        top_k_accuracy = correct / len(y_test_labels)
        
        logger.info(f"Top-{k} Accuracy: {top_k_accuracy:.4f}")
        
        return float(top_k_accuracy)
    
    def calculate_confidence_metrics(self, X_test: np.ndarray,
                                    y_test: np.ndarray) -> Dict:
        """
        Calculate confidence-related metrics.
        
        Args:
            X_test: Test images
            y_test: Test labels
        
        Returns:
            Confidence metrics
        """
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        max_proba = np.max(y_pred_proba, axis=1)
        
        if y_test.ndim > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
        
        correct = y_pred == y_test_labels
        
        metrics = {
            'mean_confidence_correct': float(max_proba[correct].mean()),
            'mean_confidence_incorrect': float(max_proba[~correct].mean()),
            'std_confidence_correct': float(max_proba[correct].std()),
            'std_confidence_incorrect': float(max_proba[~correct].std()),
            'avg_confidence': float(max_proba.mean())
        }
        
        return metrics


class ModelComparison:
    """Compare multiple models."""
    
    @staticmethod
    def compare_models(models: Dict[str, tf.keras.Model],
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      class_names: List[str] = None) -> Dict:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of models
            X_test: Test images
            y_test: Test labels
            class_names: Class names
        
        Returns:
            Comparison results
        """
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            evaluator = ModelEvaluator(model, class_names)
            metrics = evaluator.evaluate(X_test, y_test)
            
            results[model_name] = metrics
        
        return results
    
    @staticmethod
    def create_comparison_table(results: Dict) -> Dict:
        """
        Create comparison table from results.
        
        Args:
            results: Results from comparison
        
        Returns:
            Table data
        """
        table = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        
        for model_name, metrics in results.items():
            table['Model'].append(model_name)
            table['Accuracy'].append(f"{metrics['accuracy']:.4f}")
            table['Precision'].append(f"{metrics['precision']:.4f}")
            table['Recall'].append(f"{metrics['recall']:.4f}")
            table['F1-Score'].append(f"{metrics['f1_score']:.4f}")
        
        return table


class PredictionAnalysis:
    """Analyze prediction outputs."""
    
    @staticmethod
    def analyze_predictions(y_pred_proba: np.ndarray,
                           y_true: np.ndarray) -> Dict:
        """
        Analyze prediction probability distributions.
        
        Args:
            y_pred_proba: Prediction probabilities
            y_true: True labels
        
        Returns:
            Analysis results
        """
        y_pred = np.argmax(y_pred_proba, axis=1)
        max_proba = np.max(y_pred_proba, axis=1)
        
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        correct = y_pred == y_true
        
        analysis = {
            'total_samples': len(y_true),
            'correct_predictions': int(correct.sum()),
            'incorrect_predictions': int((~correct).sum()),
            'accuracy': float(correct.mean()),
            'avg_confidence': float(max_proba.mean()),
            'min_confidence': float(max_proba.min()),
            'max_confidence': float(max_proba.max()),
            'confidence_std': float(max_proba.std())
        }
        
        return analysis


def evaluate_model(model_path: str,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  class_names: List[str] = None) -> Dict:
    """
    Complete model evaluation.
    
    Args:
        model_path: Path to saved model
        X_test: Test images
        y_test: Test labels
        class_names: Class names
    
    Returns:
        Evaluation results
    """
    model = tf.keras.models.load_model(model_path)
    evaluator = ModelEvaluator(model, class_names)
    
    metrics = evaluator.evaluate(X_test, y_test)
    
    logger.info("\n" + evaluator.get_classification_report(X_test, y_test))
    
    return metrics


if __name__ == "__main__":
    # Example usage
    pass
