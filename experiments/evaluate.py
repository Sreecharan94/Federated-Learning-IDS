# experiments/evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf # Import tensorflow first
import keras
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import logging
from typing import Dict, List, Tuple, Any, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config_path: str = "./configs/config.yaml", model: Optional[keras.Model] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config.get('monitoring', {}).get('plot_dir', './outputs/evaluation/')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize label mapping
        self.label_mapping: Optional[Dict[int, str]] = {}
        self.reverse_label_mapping: Optional[Dict[str, int]] = {}
        self.model = model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                      label_mapping: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logger.info("Starting model evaluation...")

        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_data or ensure model is set before evaluating.")

        if self.model is None:
            raise ValueError("Model is not initialized. Please provide a model during initialization or load one using run_complete_evaluation.")

        if label_mapping is not None:
            self.label_mapping = label_mapping
            self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        else:
            self.label_mapping = {}
            self.reverse_label_mapping = {}

        # Make predictions
        logger.info("Making predictions...")
        assert isinstance(self.model, keras.Model), "Model must be a keras.Model instance"
        y_pred_proba = self.model.predict(X_test, verbose='0')
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate basic metrics
        logger.info("Calculating metrics...")
        accuracy = float(np.mean(y_pred == y_test))

        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=list(self.label_mapping.values()) if self.label_mapping else None,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        precision_np, recall_np, f1_np, support_np = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=np.unique(y_test)
        )

        # Overall metrics (weighted average)
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', labels=np.unique(y_test)
        )

        # Calculate ROC-AUC for each class (multiclass)
        roc_auc_scores = {}
        n_classes = y_pred_proba.shape[1]
        for i in range(n_classes):
            if len(np.unique(y_test == i)) > 1: # Only calculate if both classes present
                fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
                roc_auc_scores[i] = auc(fpr, tpr)
            else:
                roc_auc_scores[i] = 0.0 # Default if class not present in test set

        # Calculate Precision-Recall AUC
        pr_auc_scores = {}
        for i in range(n_classes):
            if len(np.unique(y_test == i)) > 1: # Only calculate if both classes present
                pr_auc = average_precision_score((y_test == i).astype(int), y_pred_proba[:, i])
                pr_auc_scores[i] = pr_auc
            else:
                pr_auc_scores[i] = 0.0 # Default if class not present

        # Prepare results dictionary with correct types - FIX astype calls
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                # Ensure they are numpy arrays before calling astype
                'precision': precision_np.astype(float).tolist() if isinstance(precision_np, np.ndarray) else [float(precision_np)] if precision_np is not None else [0.0],
                'recall': recall_np.astype(float).tolist() if isinstance(recall_np, np.ndarray) else [float(recall_np)] if recall_np is not None else [0.0],
                'f1_score': f1_np.astype(float).tolist() if isinstance(f1_np, np.ndarray) else [float(f1_np)] if f1_np is not None else [0.0],
                'support': support_np.astype(int).tolist() if isinstance(support_np, np.ndarray) else [int(support_np)] if support_np is not None else [0]
            },
            'overall_metrics': {
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1_score': float(overall_f1)
            },
            'roc_auc_scores': {k: float(v) for k, v in roc_auc_scores.items()},
            'pr_auc_scores': {k: float(v) for k, v in pr_auc_scores.items()},
            'predictions': y_pred.astype(int).tolist(),
            'probabilities': y_pred_proba.astype(float).tolist(),
            'ground_truth': y_test.astype(int).tolist()
        }

        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return results

    # ... (other methods remain the same until plotting methods)

    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       class_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot ROC curves for each class"""
        n_classes = y_pred_proba.shape[1]

        plt.figure(figsize=(10, 8))

        # Use a colormap for distinct colors - FIX Set3 access
        cmap = None
        try:
            # Attempt to get Set3 colormap
            cmap = cm.get_cmap('Set3')
            colors = cmap(np.linspace(0, 1, n_classes))
        except (AttributeError, ValueError):
            # If Set3 is not found, use a fallback or construct one
            # Example: Create a colormap with distinct colors
            # This is a simplified fallback using basic colors if Set3 is unavailable
            colors_basic = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            # Extend if n_classes > 10
            extended_colors = colors_basic * ((n_classes // len(colors_basic)) + 1)
            colors = extended_colors[:n_classes]

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
            # Use the color list or cmap
            color = colors[i] if isinstance(colors, list) else (cmap(i / (n_classes - 1)) if cmap else colors[i])
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")

        plt.show()

    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   class_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot Precision-Recall curves for each class"""
        n_classes = y_pred_proba.shape[1]

        plt.figure(figsize=(10, 8))

        # Use a colormap for distinct colors - FIX Set3 access (similar to above)
        cmap = None
        try:
            cmap = cm.get_cmap('Set3')
            colors = cmap(np.linspace(0, 1, n_classes))
        except (AttributeError, ValueError):
            colors_basic = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            extended_colors = colors_basic * ((n_classes // len(colors_basic)) + 1)
            colors = extended_colors[:n_classes]

        for i in range(n_classes):
            precision_vals, recall_vals, _ = precision_recall_curve(
                (y_true == i).astype(int), y_pred_proba[:, i]
            )
            avg_precision = average_precision_score((y_true == i).astype(int), y_pred_proba[:, i])

            class_name = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
            color = colors[i] if isinstance(colors, list) else (cmap(i / (n_classes - 1)) if cmap else colors[i])
            plt.plot(recall_vals, precision_vals, color=color, lw=2,
                    label=f'{class_name} (AP = {avg_precision:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Multi-class Classification')
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")

        plt.show()

    def run_complete_evaluation(self, X_test: np.ndarray, y_test: np.ndarray,
                               label_mapping: Optional[Dict[int, str]] = None,
                               model_path: Optional[str] = None) -> tuple:
        """Run complete evaluation and return results and report"""
        if model_path and self.model is None:
            self.model = keras.models.load_model(model_path)
        
        results = self.evaluate_model(X_test, y_test, label_mapping)
        report = results.get('classification_report', {})
        
        return results, report

# Test function - ensure tf.keras usage
def test_evaluation():
    """Test the evaluation functionality"""
    # Create dummy data
    X_test = np.random.random((200, 50, 78)).astype(np.float32)
    y_test = np.random.randint(0, 10, 200).astype(np.int32)

    # Create simple model for testing - USE tf.keras
    inputs = keras.Input(shape=(50, 78)) # Use keras.Input
    x = keras.layers.LSTM(64, return_sequences=True)(inputs) # Use keras.layers
    x = keras.layers.GlobalAveragePooling1D()(x) # Use keras.layers
    outputs = keras.layers.Dense(10, activation='softmax')(x) # Use keras.layers
    model = keras.Model(inputs, outputs) # Use keras.Model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Save test model
    test_model_path = "./test_model.h5"
    model.save(test_model_path)

    # Create evaluator
    evaluator = ModelEvaluator(model=model)

    # Create label mapping
    label_mapping = {i: f"Attack_{i}" for i in range(10)}

    # Run evaluation
    results, report = evaluator.run_complete_evaluation(
        X_test, y_test, label_mapping, test_model_path
    )

    print(f"Evaluation completed with accuracy: {results['accuracy']:.4f}")

    # Clean up
    if os.path.exists(test_model_path):
        os.remove(test_model_path)

    print("Evaluation test passed!")

if __name__ == "__main__":
    test_evaluation()