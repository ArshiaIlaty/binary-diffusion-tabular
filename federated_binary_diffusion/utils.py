import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Dict, List, Tuple

class FederatedLogger:
    def __init__(self, log_dir: str = "./results"):
        """Initialize logger with directory for saving results"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            'client1': {'loss': [], 'accuracy': []},
            'client2': {'loss': [], 'accuracy': []},
            'global': {'loss': [], 'accuracy': []}
        }
        
    def log_metrics(self, round_num: int, client_id: str, loss: float, accuracy: float):
        """Log metrics for a specific client or global model"""
        self.metrics[client_id]['loss'].append(loss)
        self.metrics[client_id]['accuracy'].append(accuracy)
        
    def plot_training_curves(self):
        """Plot training curves for loss and accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss curves
        for client_id in self.metrics:
            ax1.plot(self.metrics[client_id]['loss'], 
                    label=f"{client_id} Loss",
                    marker='o')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy curves
        for client_id in self.metrics:
            ax2.plot(self.metrics[client_id]['accuracy'], 
                    label=f"{client_id} Accuracy",
                    marker='o')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()

def compare_data_distributions(original_data: pd.DataFrame, 
                             client1_data: pd.DataFrame, 
                             client2_data: pd.DataFrame,
                             save_dir: str = "./results"):
    """Compare and visualize data distributions across datasets"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select numerical columns for visualization
    num_cols = original_data.select_dtypes(include=[np.number]).columns[:4]  # First 4 numerical columns
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, col in enumerate(num_cols):
        sns.kdeplot(data=original_data, x=col, label='Original', ax=axes[idx])
        sns.kdeplot(data=client1_data, x=col, label='Client 1', ax=axes[idx])
        sns.kdeplot(data=client2_data, x=col, label='Client 2', ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_distributions.png'))
    plt.close()

def plot_roc_curves(y_true: np.ndarray, 
                   y_pred_proba: Dict[str, np.ndarray],
                   save_dir: str = "./results"):
    """Plot ROC curves for different models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, y_pred in y_pred_proba.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()

def plot_confusion_matrices(y_true: np.ndarray,
                          y_pred: Dict[str, np.ndarray],
                          save_dir: str = "./results"):
    """Plot confusion matrices for different models"""
    n_models = len(y_pred)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, pred) in zip(axes, y_pred.items()):
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(f'{model_name}\nConfusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.close()

def plot_model_comparison(metrics: Dict[str, Dict[str, float]], 
                         save_dir: str = "./results"):
    """Plot comparison of different metrics across models"""
    metric_names = list(next(iter(metrics.values())).keys())
    n_metrics = len(metric_names)
    model_names = list(metrics.keys())
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_names):
        values = [metrics[model][metric] for model in model_names]
        ax.bar(model_names, values)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.grid(True)
        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()

def save_metrics_report(metrics: Dict[str, Dict[str, float]], 
                       save_dir: str = "./results"):
    """Save detailed metrics report as CSV"""
    df = pd.DataFrame(metrics).round(4)
    df.to_csv(os.path.join(save_dir, 'metrics_report.csv'))
    
    # Also create a formatted text report
    with open(os.path.join(save_dir, 'metrics_report.txt'), 'w') as f:
        f.write("Model Performance Comparison Report\n")
        f.write("=================================\n\n")
        for model, model_metrics in metrics.items():
            f.write(f"{model}:\n")
            for metric, value in model_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
