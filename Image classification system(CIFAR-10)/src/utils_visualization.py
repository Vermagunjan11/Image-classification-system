"""
Visualization Utilities for Model Performance and Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


class MetricsVisualizer:
    """Visualize model metrics and performance."""
    
    @staticmethod
    def plot_history(history: dict,
                    metrics: List[str] = None,
                    title: str = "Model Performance") -> go.Figure:
        """
        Plot training history using Plotly.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            title: Plot title
        
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ['accuracy', 'loss']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[m.capitalize() for m in metrics]
        )
        
        epochs = range(1, len(history.get(metrics[0], [])) + 1)
        
        for idx, metric in enumerate(metrics, 1):
            if metric in history:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=history[metric],
                             name=f"Train {metric}",
                             mode='lines+markers',
                             line=dict(width=2)),
                    row=1, col=idx
                )
            
            val_metric = f'val_{metric}'
            if val_metric in history:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=history[val_metric],
                             name=f"Val {metric}",
                             mode='lines+markers',
                             line=dict(width=2, dash='dash')),
                    row=1, col=idx
                )
        
        fig.update_layout(
            title_text=title,
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str] = None) -> go.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
        
        Returns:
            Plotly figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray,
                      y_score: np.ndarray) -> go.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels (binary)
            y_score: Predicted scores
        
        Returns:
            Plotly figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            mode='lines',
            line=dict(width=2, color='#1f77b4')
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(width=2, color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(models: Dict[str, dict]) -> go.Figure:
        """
        Compare metrics across models.
        
        Args:
            models: Dictionary of model metrics
                   {'model_name': {'accuracy': 0.9, 'precision': 0.92, ...}}
        
        Returns:
            Plotly figure
        """
        metrics = set()
        for model_metrics in models.values():
            metrics.update(model_metrics.keys())
        
        metrics = sorted(list(metrics))
        
        fig = go.Figure()
        
        for model_name, model_metrics in models.items():
            values = [model_metrics.get(m, 0) for m in metrics]
            fig.add_trace(go.Bar(
                x=metrics,
                y=values,
                name=model_name
            ))
        
        fig.update_layout(
            title='Model Metrics Comparison',
            barmode='group',
            xaxis_title='Metrics',
            yaxis_title='Score',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_prediction_distribution(predictions: np.ndarray,
                                    true_labels: np.ndarray = None,
                                    title: str = "Prediction Distribution") -> go.Figure:
        """
        Plot distribution of predictions.
        
        Args:
            predictions: Array of predictions
            true_labels: Optional true labels
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=predictions,
            nbinsx=50,
            name='Predictions',
            opacity=0.7
        ))
        
        if true_labels is not None:
            fig.add_trace(go.Histogram(
                x=predictions[true_labels == 1],
                nbinsx=50,
                name='Correct',
                opacity=0.7
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Prediction Score',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        
        return fig


class PredictionVisualizer:
    """Visualize predictions and results."""
    
    @staticmethod
    def plot_top_predictions(predictions: np.ndarray,
                            class_names: List[str],
                            top_k: int = 5,
                            title: str = "Top Predictions") -> go.Figure:
        """
        Plot top K predictions as bar chart.
        
        Args:
            predictions: Prediction array
            class_names: Class names
            top_k: Number of top predictions
            title: Plot title
        
        Returns:
            Plotly figure
        """
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_scores = predictions[top_indices]
        
        fig = go.Figure(data=[
            go.Bar(x=top_classes, y=top_scores,
                  marker_color='lightblue')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Class',
            yaxis_title='Confidence',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_class_distribution(labels: np.ndarray,
                               class_names: List[str],
                               title: str = "Class Distribution") -> go.Figure:
        """
        Plot class distribution.
        
        Args:
            labels: Array of labels
            class_names: Class names
            title: Plot title
        
        Returns:
            Plotly figure
        """
        unique, counts = np.unique(labels, return_counts=True)
        class_names_used = [class_names[i] for i in unique]
        
        fig = go.Figure(data=[
            go.Bar(x=class_names_used, y=counts,
                  marker_color='indianred')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Class',
            yaxis_title='Count',
            height=400
        )
        
        return fig


class ModelComparisonVisualizer:
    """Visualize model comparisons."""
    
    @staticmethod
    def create_comparison_table(models: Dict[str, dict]) -> dict:
        """
        Create comparison table data.
        
        Args:
            models: Dictionary of model metrics
        
        Returns:
            Table data
        """
        columns = ['Model'] + list(next(iter(models.values())).keys())
        data = []
        
        for model_name, metrics in models.items():
            row = [model_name] + [metrics.get(col, '-') for col in columns[1:]]
            data.append(row)
        
        return {'columns': columns, 'data': data}
    
    @staticmethod
    def plot_radar_comparison(models: Dict[str, dict],
                             metrics: List[str] = None) -> go.Figure:
        """
        Create radar chart for model comparison.
        
        Args:
            models: Dictionary of model metrics
            metrics: Metrics to compare
        
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = list(next(iter(models.values())).keys())
        
        fig = go.Figure()
        
        for model_name, model_metrics in models.items():
            values = [model_metrics.get(m, 0) for m in metrics]
            values += values[:1]  # Complete the circle
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600
        )
        
        return fig


def plot_image_grid(images: List[np.ndarray],
                   titles: List[str] = None,
                   cols: int = 3) -> plt.Figure:
    """
    Plot grid of images.
    
    Args:
        images: List of images
        titles: List of titles
        cols: Number of columns
    
    Returns:
        Matplotlib figure
    """
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image)
        ax.axis('off')
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
    
    # Hide unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')
    
    plt.tight_layout()
    return fig
