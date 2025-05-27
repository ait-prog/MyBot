import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralNetworkVisualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.colors = sns.color_palette("husl", 8)
        plt.style.use('seaborn')

    def plot_training_history(
            self,
            history: Dict,
            metrics: List[str] = None,
            save_path: Optional[str] = None
    ):
        """
        Visualize training curves

        Args:
            history: Training history
            metrics: Metrics to visualize
            save_path: Path to save plot
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        for i, metric in enumerate(metrics):
            ax = axes[i] if n_metrics > 1 else axes
            ax.plot(history[f'{metric}'], label=f'Training {metric}', color=self.colors[0])
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric}', color=self.colors[1])
            ax.set_title(f'{metric.capitalize()} over time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()

    def plot_confusion_matrix(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_names: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """
        Visualize confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()

    def plot_roc_curves(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_names: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """
        Visualize ROC curves

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            class_names: Class names
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))

        for i in range(y_true.shape[1]):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            label = f'{class_names[i]} (AUC = {roc_auc:.2f})' if class_names else f'Class {i} (AUC = {roc_auc:.2f})'
            plt.plot(fpr, tpr, label=label, color=self.colors[i % len(self.colors)])

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curves plot saved to {save_path}")
        else:
            plt.show()

    def plot_feature_importance(
            self,
            feature_names: List[str],
            importance_scores: np.ndarray,
            save_path: Optional[str] = None
    ):
        """
        Visualize feature importance

        Args:
            feature_names: Feature names
            importance_scores: Importance scores
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', data=importance_df, palette=self.colors)
        plt.title('Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

    def plot_attention_weights(
            self,
            attention_weights: np.ndarray,
            input_tokens: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """
        Visualize attention weights

        Args:
            attention_weights: Attention weights
            input_tokens: Input tokens
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            xticklabels=input_tokens if input_tokens else 'auto',
            yticklabels=input_tokens if input_tokens else 'auto'
        )
        plt.title('Attention Weights')

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Attention weights plot saved to {save_path}")
        else:
            plt.show()

    def plot_gradients(
            self,
            gradients: List[tf.Tensor],
            layer_names: List[str],
            save_path: Optional[str] = None
    ):
        """
        Visualize gradients

        Args:
            gradients: Gradients
            layer_names: Layer names
            save_path: Path to save plot
        """
        plt.figure(figsize=(15, 5))
        for i, (grad, name) in enumerate(zip(gradients, layer_names)):
            if grad is not None:
                plt.subplot(1, len(gradients), i + 1)
                plt.hist(grad.numpy().flatten(), bins=50, color=self.colors[i % len(self.colors)])
                plt.title(f'{name} Gradients')

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Gradients plot saved to {save_path}")
        else:
            plt.show()

    def plot_feature_maps(
            self,
            feature_maps: np.ndarray,
            n_features: int = 16,
            save_path: Optional[str] = None
    ):
        """
        Visualize feature maps

        Args:
            feature_maps: Feature maps
            n_features: Number of features to visualize
            save_path: Path to save plot
        """
        plt.figure(figsize=(20, 20))
        for i in range(min(n_features, feature_maps.shape[-1])):
            plt.subplot(4, 4, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature maps plot saved to {save_path}")
        else:
            plt.show()

    def plot_decision_boundary(
            self,
            model: models.Model,
            X: np.ndarray,
            y: np.ndarray,
            save_path: Optional[str] = None
    ):
        """
        Visualize decision boundary

        Args:
            model: Model
            X: Features
            y: Labels
            save_path: Path to save plot
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary visualization requires 2D features")

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.argmax(axis=1)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, colors=self.colors)
        plt.scatter(X[:, 0], X[:, 1], c=y.argmax(axis=1), alpha=0.8, cmap='viridis')
        plt.title('Decision Boundary')

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Decision boundary plot saved to {save_path}")
        else:
            plt.show()

    def plot_model_performance(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_names: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """
        Visualize model performance

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            save_path: Path to save plot
        """
        # ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(y_true.shape[1]):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            label = f'{class_names[i]} (AUC = {roc_auc:.2f})' if class_names else f'Class {i} (AUC = {roc_auc:.2f})'
            plt.plot(fpr, tpr, label=label, color=self.colors[i % len(self.colors)])

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Model performance plot saved to {save_path}")
        else:
            plt.show()

        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
            target_names=class_names
        ))

    def plot_learning_curves(
            self,
            train_sizes: np.ndarray,
            train_scores: np.ndarray,
            val_scores: np.ndarray,
            save_path: Optional[str] = None
    ):
        """
        Visualize learning curves

        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score', color=self.colors[0])
        plt.plot(train_sizes, val_scores.mean(axis=1), label='Cross-validation score', color=self.colors[1])
        plt.fill_between(
            train_sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            alpha=0.1,
            color=self.colors[0]
        )
        plt.fill_between(
            train_sizes,
            val_scores.mean(axis=1) - val_scores.std(axis=1),
            val_scores.mean(axis=1) + val_scores.std(axis=1),
            alpha=0.1,
            color=self.colors[1]
        )
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Learning curves plot saved to {save_path}")
        else:
            plt.show()

    def plot_model_architecture(
            self,
            model: models.Model,
            save_path: Optional[str] = None
    ):
        """
        Visualize model architecture

        Args:
            model: Model to visualize
            save_path: Path to save plot
        """
        tf.keras.utils.plot_model(
            model,
            to_file=save_path if save_path else 'model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB'
        )
        logger.info(f"Model architecture plot saved to {save_path if save_path else 'model_architecture.png'}")
