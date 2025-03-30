"""
Model Evaluator Module

This module provides tools for evaluating machine learning models,
comparing performance across different versions, and generating reports.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

class ModelEvaluator:
    """
    Provides functionality for evaluating machine learning models and comparing performance
    across different versions.
    
    Features:
    - Comprehensive model evaluation
    - Performance comparison across model versions
    - Visualization of model metrics
    - Report generation
    """
    
    def __init__(self, config=None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.results_dir = os.path.join("data", "model_evaluation")
        self.reports_dir = os.path.join(self.results_dir, "reports")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.logger.info("ModelEvaluator initialized")
    
    def evaluate_classification_model(self, model, X_test, y_test, threshold=0.5):
        """
        Evaluate a classification model and return performance metrics.
        
        Args:
            model: Trained classification model
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): True labels
            threshold (float): Probability threshold for binary classification
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating classification model")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            # Handle multi-class vs binary
            if y_prob.shape[1] > 2:
                # Multi-class
                y_pred = model.predict(X_test)
            else:
                # Binary
                y_prob = y_prob[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_prob = None
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'threshold': threshold
        }
        
        # Check if binary or multi-class
        unique_classes = np.unique(y_test)
        is_binary = len(unique_classes) <= 2
        
        if is_binary:
            # Binary classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # ROC AUC (only if we have probabilities)
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                
                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Precision-Recall curve data
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
                metrics['pr_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Get true/false positives/negatives
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_positives'] = int(tp)
                metrics['false_positives'] = int(fp)
                metrics['true_negatives'] = int(tn)
                metrics['false_negatives'] = int(fn)
                
                # Calculate additional metrics
                if tp + fp > 0:
                    metrics['precision'] = tp / (tp + fp)
                else:
                    metrics['precision'] = 0
                    
                if tp + fn > 0:
                    metrics['recall'] = tp / (tp + fn)
                else:
                    metrics['recall'] = 0
                    
                if metrics['precision'] + metrics['recall'] > 0:
                    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                else:
                    metrics['f1_score'] = 0
        else:
            # Multi-class classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        self.logger.info(f"Classification model evaluated with accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def evaluate_regression_model(self, model, X_test, y_test):
        """
        Evaluate a regression model and return performance metrics.
        
        Args:
            model: Trained regression model
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): True values
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating regression model")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        # Mean absolute percentage error (MAPE)
        # Avoid division by zero
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if any(mask) else np.inf
        
        # Compiled metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'explained_variance': r2,  # For consistency with naming conventions
            'residual_stats': {
                'mean': np.mean(y_test - y_pred),
                'std': np.std(y_test - y_pred),
                'min': np.min(y_test - y_pred),
                'max': np.max(y_test - y_pred)
            }
        }
        
        self.logger.info(f"Regression model evaluated with R² score: {r2:.4f}")
        
        return metrics
    
    def compare_models(self, model_results, primary_metric='accuracy'):
        """
        Compare multiple model versions based on their evaluation metrics.
        
        Args:
            model_results (list): List of dictionaries containing model evaluation results
            primary_metric (str): The primary metric to use for comparison
            
        Returns:
            dict: Comparison results
        """
        self.logger.info(f"Comparing {len(model_results)} models based on {primary_metric}")
        
        # Check if model_results is not empty
        if not model_results:
            self.logger.warning("No models to compare")
            return None
        
        # Extract metrics for each model
        comparison = {
            'primary_metric': primary_metric,
            'models': [],
            'best_model_index': None,
            'metric_comparison': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Extract common metrics
        common_metrics = set()
        for model_result in model_results:
            common_metrics.update(model_result.keys())
        
        # Remove complex metrics that don't make sense to compare directly
        complex_metrics = ['confusion_matrix', 'classification_report', 'roc_curve', 'pr_curve', 'residual_stats']
        common_metrics = [m for m in common_metrics if m not in complex_metrics]
        
        # Organize data for comparison
        model_data = []
        for i, model_result in enumerate(model_results):
            model_info = {
                'index': i,
                'name': model_result.get('name', f'Model {i+1}'),
                'version': model_result.get('version', f'v{i+1}'),
                'metrics': {metric: model_result.get(metric) for metric in common_metrics if metric in model_result}
            }
            model_data.append(model_info)
        
        comparison['models'] = model_data
        
        # Find best model based on primary metric
        if primary_metric in common_metrics:
            # Check if higher is better for this metric
            higher_is_better = primary_metric in [
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'r2', 'explained_variance'
            ]
            
            # Extract metric values 
            metric_values = [
                (i, model['metrics'].get(primary_metric)) 
                for i, model in enumerate(model_data) 
                if primary_metric in model['metrics'] and model['metrics'][primary_metric] is not None
            ]
            
            if metric_values:
                if higher_is_better:
                    best_idx, _ = max(metric_values, key=lambda x: x[1])
                else:
                    best_idx, _ = min(metric_values, key=lambda x: x[1])
                
                comparison['best_model_index'] = best_idx
                
                self.logger.info(f"Best model: {model_data[best_idx]['name']} (index {best_idx})")
            
        # Compare each metric across models
        for metric in common_metrics:
            comparison['metric_comparison'][metric] = {
                'values': [model['metrics'].get(metric) for model in model_data]
            }
            
            # Calculate improvement if applicable
            if len(model_data) > 1 and all(isinstance(model['metrics'].get(metric), (int, float)) for model in model_data if metric in model['metrics']):
                comparison['metric_comparison'][metric]['mean'] = np.mean([model['metrics'].get(metric) for model in model_data if metric in model['metrics']])
                comparison['metric_comparison'][metric]['std'] = np.std([model['metrics'].get(metric) for model in model_data if metric in model['metrics']])
        
        return comparison
    
    def visualize_model_comparison(self, comparison, output_file=None):
        """
        Generate visualizations for model comparison.
        
        Args:
            comparison (dict): Model comparison results
            output_file (str, optional): Path to save the visualization
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Visualizing model comparison")
        
        try:
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 14))
            
            # Get model names for x-axis
            model_names = [model.get('name', f"Model {i+1}") for i, model in enumerate(comparison['models'])]
            
            # Determine if we're looking at classification or regression metrics
            classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            regression_metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
            
            metric_comparison = comparison['metric_comparison']
            
            primary_metric = comparison['primary_metric']
            best_model_index = comparison['best_model_index']
            
            if primary_metric in classification_metrics:
                model_type = 'classification'
            elif primary_metric in regression_metrics:
                model_type = 'regression'
            else:
                model_type = 'unknown'
            
            # Plot primary metric comparison
            if primary_metric in metric_comparison:
                metric_values = metric_comparison[primary_metric]['values']
                
                # Create bar chart
                bars = axs[0, 0].bar(model_names, metric_values)
                
                # Highlight best model
                if best_model_index is not None and 0 <= best_model_index < len(bars):
                    bars[best_model_index].set_color('green')
                
                axs[0, 0].set_title(f'{primary_metric.capitalize()} Comparison', fontsize=14)
                axs[0, 0].set_xlabel('Model')
                axs[0, 0].set_ylabel(primary_metric.capitalize())
                
                # Add value labels
                for i, v in enumerate(metric_values):
                    axs[0, 0].text(i, v + 0.02 * max(metric_values), f"{v:.4f}", ha='center')
                
                # Set y-axis limits for better visualization
                if primary_metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                    axs[0, 0].set_ylim(0, 1.1)
            else:
                axs[0, 0].text(0.5, 0.5, f"No data for {primary_metric}", 
                             ha='center', va='center', transform=axs[0, 0].transAxes)
            
            # Plot secondary metrics comparison (based on model type)
            if model_type == 'classification':
                secondary_metrics = [m for m in ['precision', 'recall', 'f1_score'] if m in metric_comparison and m != primary_metric]
            else:  # regression
                secondary_metrics = [m for m in ['rmse', 'mae', 'r2'] if m in metric_comparison and m != primary_metric]
            
            if secondary_metrics:
                # Create grouped bar chart
                width = 0.8 / len(secondary_metrics)
                offsets = np.linspace(-(len(secondary_metrics) - 1) * width / 2, (len(secondary_metrics) - 1) * width / 2, len(secondary_metrics))
                
                for i, metric in enumerate(secondary_metrics):
                    metric_values = metric_comparison[metric]['values']
                    x_pos = np.arange(len(model_names)) + offsets[i]
                    axs[0, 1].bar(x_pos, metric_values, width=width, label=metric.upper())
                
                axs[0, 1].set_title('Secondary Metrics Comparison', fontsize=14)
                axs[0, 1].set_xlabel('Model')
                axs[0, 1].set_ylabel('Metric Value')
                axs[0, 1].set_xticks(np.arange(len(model_names)))
                axs[0, 1].set_xticklabels(model_names)
                axs[0, 1].legend()
            else:
                axs[0, 1].text(0.5, 0.5, "No secondary metrics available", 
                             ha='center', va='center', transform=axs[0, 1].transAxes)
            
            # Plot metric heatmap
            if metric_comparison:
                # Create data for heatmap
                metrics_to_plot = [m for m in metric_comparison if all(v is not None for v in metric_comparison[m]['values'])]
                
                if metrics_to_plot:
                    heatmap_data = pd.DataFrame(
                        {metric: metric_comparison[metric]['values'] for metric in metrics_to_plot},
                        index=model_names
                    )
                    
                    # Normalize data for better visualization
                    normalized_data = heatmap_data.copy()
                    for col in normalized_data.columns:
                        if col in ['mse', 'rmse', 'mae', 'mape']:  # Lower is better
                            normalized_data[col] = 1 - (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())
                        else:  # Higher is better
                            normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())
                    
                    # Create heatmap
                    sns.heatmap(normalized_data, annot=heatmap_data, fmt=".4f", cmap="YlGnBu", ax=axs[1, 0])
                    axs[1, 0].set_title('Metrics Comparison Heatmap', fontsize=14)
                else:
                    axs[1, 0].text(0.5, 0.5, "Insufficient data for heatmap", 
                                 ha='center', va='center', transform=axs[1, 0].transAxes)
            else:
                axs[1, 0].text(0.5, 0.5, "No metric comparison data available", 
                             ha='center', va='center', transform=axs[1, 0].transAxes)
                
            # Performance trends if there's version information
            if all('version' in model for model in comparison['models']):
                # Try to parse versions as numbers (if they're in the format v1, v2, etc.)
                versions = []
                for model in comparison['models']:
                    version_str = model['version']
                    try:
                        # Extract numeric part
                        if version_str.startswith('v'):
                            version_num = float(version_str[1:])
                        else:
                            version_num = float(version_str)
                        versions.append(version_num)
                    except ValueError:
                        # Use index as fallback
                        versions.append(model['index'])
                
                # Sort models by version
                sorted_indices = np.argsort(versions)
                sorted_versions = [versions[i] for i in sorted_indices]
                sorted_metric_values = [metric_comparison[primary_metric]['values'][i] for i in sorted_indices]
                
                # Plot trend
                axs[1, 1].plot(sorted_versions, sorted_metric_values, 'o-', linewidth=2)
                axs[1, 1].set_title(f'{primary_metric.capitalize()} Trend Across Versions', fontsize=14)
                axs[1, 1].set_xlabel('Version')
                axs[1, 1].set_ylabel(primary_metric.capitalize())
                axs[1, 1].grid(True)
            else:
                axs[1, 1].text(0.5, 0.5, "Version information not available", 
                             ha='center', va='center', transform=axs[1, 1].transAxes)
            
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved model comparison visualization to {output_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing model comparison: {str(e)}")
            return None, None
    
    def generate_evaluation_report(self, model_results, model_name, model_version, output_file=None):
        """
        Generate a comprehensive evaluation report for a model.
        
        Args:
            model_results (dict): Model evaluation results
            model_name (str): Name of the model
            model_version (str): Version of the model
            output_file (str, optional): Path to save the report
            
        Returns:
            dict: Report data
        """
        self.logger.info(f"Generating evaluation report for {model_name} {model_version}")
        
        report = {
            'model_name': model_name,
            'model_version': model_version,
            'report_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {},
            'visualizations': {},
            'feature_importance': {},
            'recommendations': []
        }
        
        # Copy metrics
        for key, value in model_results.items():
            if key not in ['name', 'version']:
                report['metrics'][key] = value
        
        # Generate recommendations based on metrics
        if 'accuracy' in model_results:
            # Classification model
            accuracy = model_results['accuracy']
            
            if accuracy < 0.6:
                report['recommendations'].append("Model accuracy is low. Consider feature engineering, gathering more data, or trying different algorithms.")
            elif accuracy < 0.8:
                report['recommendations'].append("Model accuracy is moderate. Consider hyperparameter tuning or adding more relevant features.")
            
            if 'precision' in model_results and 'recall' in model_results:
                precision = model_results['precision']
                recall = model_results['recall']
                
                if precision < 0.6 and recall >= 0.8:
                    report['recommendations'].append("Model has high recall but low precision. Consider adjusting the decision threshold or using a different loss function.")
                elif precision >= 0.8 and recall < 0.6:
                    report['recommendations'].append("Model has high precision but low recall. Consider adjusting the decision threshold to capture more positive cases.")
        
        elif 'r2' in model_results:
            # Regression model
            r2 = model_results['r2']
            
            if r2 < 0.3:
                report['recommendations'].append("Model R² score is very low. Consider adding more relevant features or trying different algorithms.")
            elif r2 < 0.6:
                report['recommendations'].append("Model R² score is moderate. Consider hyperparameter tuning or feature engineering.")
        
        # Save report
        if output_file:
            report_dir = os.path.dirname(output_file)
            os.makedirs(report_dir, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Saved evaluation report to {output_file}")
        
        return report
    
    def visualize_confusion_matrix(self, confusion_matrix, class_names=None):
        """
        Visualize a confusion matrix.
        
        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix
            class_names (list, optional): Names of the classes
            
        Returns:
            tuple: Figure and axes
        """
        # Convert to numpy array if it's a list
        if isinstance(confusion_matrix, list):
            confusion_matrix = np.array(confusion_matrix)
        
        # Create default class names if not provided
        if class_names is None:
            class_names = [f"Class {i}" for i in range(confusion_matrix.shape[0])]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set tick labels
        ax.set(xticks=np.arange(confusion_matrix.shape[1]),
               yticks=np.arange(confusion_matrix.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        thresh = confusion_matrix.max() / 2.0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.title('Confusion Matrix')
        
        return fig, ax
    
    def visualize_roc_curve(self, fpr, tpr, roc_auc):
        """
        Visualize a ROC curve.
        
        Args:
            fpr (array-like): False positive rates
            tpr (array-like): True positive rates
            roc_auc (float): ROC AUC score
            
        Returns:
            tuple: Figure and axes
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        
        return fig, ax
    
    def visualize_precision_recall_curve(self, precision, recall):
        """
        Visualize a precision-recall curve.
        
        Args:
            precision (array-like): Precision values
            recall (array-like): Recall values
            
        Returns:
            tuple: Figure and axes
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot precision-recall curve
        ax.plot(recall, precision, color='blue', lw=2)
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        
        # Add F1 score contours
        x = np.linspace(0.01, 1, 100)
        for f1 in np.linspace(0.2, 0.8, 4):
            y = (f1 * x) / (2 * x - f1)
            ax.plot(x, y, '--', color='gray', alpha=0.5, linewidth=1)
            # Find a suitable position for the label
            y_pos = (f1 * 0.5) / (2 * 0.5 - f1) if 2 * 0.5 - f1 > 0 else 0.9
            ax.annotate(f'F1={f1:.1f}', xy=(0.5, y_pos), color='gray')
            
        # Calculate average precision
        ap = np.sum((recall[:-1] - recall[1:]) * precision[1:])
        ax.legend([f'AP={ap:.4f}'])
        
        return fig, ax
    
    def visualize_residuals(self, y_true, y_pred):
        """
        Visualize regression model residuals.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            tuple: Figure and axes
        """
        residuals = y_true - y_pred
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Residuals vs predicted values
        axs[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axs[0, 0].axhline(y=0, color='r', linestyle='--')
        axs[0, 0].set_xlabel('Predicted values')
        axs[0, 0].set_ylabel('Residuals')
        axs[0, 0].set_title('Residuals vs Predicted Values')
        
        # Histogram of residuals
        axs[0, 1].hist(residuals, bins=30, alpha=0.7, color='blue')
        axs[0, 1].axvline(x=0, color='r', linestyle='--')
        axs[0, 1].set_xlabel('Residuals')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Residuals Distribution')
        
        # Predicted vs actual values
        axs[1, 0].scatter(y_true, y_pred, alpha=0.5)
        # Add diagonal line (perfect predictions)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axs[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axs[1, 0].set_xlabel('True values')
        axs[1, 0].set_ylabel('Predicted values')
        axs[1, 0].set_title('Predicted vs True Values')
        
        # Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axs[1, 1])
        axs[1, 1].set_title('Q-Q Plot of Residuals')
        
        fig.tight_layout()
        
        return fig, axs
    
    def save_evaluation_results(self, metrics, model_name, model_version, is_classification=True):
        """
        Save model evaluation results to a file.
        
        Args:
            metrics (dict): Evaluation metrics
            model_name (str): Name of the model
            model_version (str): Version of the model
            is_classification (bool): Whether this is a classification model
            
        Returns:
            str: Path to the saved file
        """
        # Create a results dictionary with metadata
        results = {
            'name': model_name,
            'version': model_version,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'classification' if is_classification else 'regression',
            **metrics
        }
        
        # Create filename
        safe_name = model_name.replace(' ', '_').lower()
        safe_version = model_version.replace('.', '_').replace(' ', '_').lower()
        filename = f"{safe_name}_{safe_version}_evaluation.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {filepath}")
        
        return filepath
    
    def load_evaluation_results(self, filepath):
        """
        Load model evaluation results from a file.
        
        Args:
            filepath (str): Path to the results file
            
        Returns:
            dict: Evaluation results
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.logger.info(f"Loaded evaluation results from {filepath}")
            return results
        except Exception as e:
            self.logger.error(f"Error loading evaluation results: {str(e)}")
            return None
    
    def get_all_evaluations(self, model_name=None):
        """
        Get all saved evaluation results.
        
        Args:
            model_name (str, optional): Filter by model name
            
        Returns:
            list: List of evaluation result dictionaries
        """
        # List all evaluation files
        eval_files = [f for f in os.listdir(self.results_dir) if f.endswith('_evaluation.json')]
        
        # Filter by model name if specified
        if model_name:
            safe_name = model_name.replace(' ', '_').lower()
            eval_files = [f for f in eval_files if f.startswith(safe_name)]
        
        # Load each file
        evaluations = []
        for filename in eval_files:
            filepath = os.path.join(self.results_dir, filename)
            result = self.load_evaluation_results(filepath)
            if result:
                evaluations.append(result)
        
        return evaluations
    
    def generate_model_card(self, model_results, feature_importance=None, model_info=None, output_file=None):
        """
        Generate a model card with key information.
        
        Args:
            model_results (dict): Model evaluation results
            feature_importance (dict, optional): Feature importance data
            model_info (dict, optional): Additional model information
            output_file (str, optional): Path to save the model card
            
        Returns:
            str: Model card markdown
        """
        self.logger.info("Generating model card")
        
        # Get model metadata
        model_name = model_results.get('name', 'Unnamed Model')
        model_version = model_results.get('version', 'Unknown Version')
        model_type = model_results.get('model_type', 'Unknown Type')
        timestamp = model_results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Start building the markdown
        md = f"# Model Card: {model_name} v{model_version}\n\n"
        md += f"**Generated:** {timestamp}\n\n"
        
        # Model Overview
        md += "## Model Overview\n\n"
        if model_info:
            if 'description' in model_info:
                md += f"{model_info['description']}\n\n"
            
            if 'purpose' in model_info:
                md += f"**Purpose:** {model_info['purpose']}\n\n"
            
            if 'training_data' in model_info:
                md += f"**Training Data:** {model_info['training_data']}\n\n"
            
            if 'algorithm' in model_info:
                md += f"**Algorithm:** {model_info['algorithm']}\n\n"
        else:
            md += f"A {model_type} model used for prediction.\n\n"
        
        # Performance Metrics
        md += "## Performance Metrics\n\n"
        
        if model_type == 'classification':
            metrics_to_report = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            md += "| Metric | Value |\n| --- | --- |\n"
            
            for metric in metrics_to_report:
                if metric in model_results:
                    md += f"| {metric.replace('_', ' ').title()} | {model_results[metric]:.4f} |\n"
                    
            # Confusion Matrix
            if 'confusion_matrix' in model_results:
                md += "\n### Confusion Matrix\n\n"
                cm = model_results['confusion_matrix']
                
                if isinstance(cm, list):
                    md += "```\n"
                    for row in cm:
                        md += "| " + " | ".join([str(x) for x in row]) + " |\n"
                    md += "```\n\n"
        
        elif model_type == 'regression':
            metrics_to_report = ['mse', 'rmse', 'mae', 'r2', 'mape']
            md += "| Metric | Value |\n| --- | --- |\n"
            
            for metric in metrics_to_report:
                if metric in model_results:
                    md += f"| {metric.upper()} | {model_results[metric]:.4f} |\n"
        
        # Feature Importance
        if feature_importance:
            md += "\n## Feature Importance\n\n"
            md += "| Feature | Importance |\n| --- | --- |\n"
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features[:10]:  # Top 10 features
                md += f"| {feature} | {importance:.4f} |\n"
        
        # Recommendations
        if 'recommendations' in model_results:
            md += "\n## Recommendations\n\n"
            for recommendation in model_results['recommendations']:
                md += f"- {recommendation}\n"
        
        # Save model card
        if output_file:
            with open(output_file, 'w') as f:
                f.write(md)
            
            self.logger.info(f"Saved model card to {output_file}")
        
        return md
