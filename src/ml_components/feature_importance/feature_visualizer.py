"""
Feature Visualizer Module

This module provides tools for visualizing feature importance and relationships in machine learning models,
including bar charts, heatmaps, partial dependence plots, and SHAP value visualizations.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.inspection import plot_partial_dependence

# Try to import SHAP - an optional dependency
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class FeatureVisualizer:
    """
    Provides methods for visualizing feature importance and relationships in machine learning models.
    
    Features:
    - Feature importance bar charts and heatmaps
    - SHAP value visualization (summary, dependency, force plots)
    - Partial dependence plots
    - Feature correlation heatmaps
    - Time-based feature importance visualization
    """
    
    def __init__(self, config=None):
        """
        Initialize the FeatureVisualizer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.plots_dir = os.path.join("data", "feature_importance", "visualizations")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set default plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Flag to indicate if SHAP is available
        self.shap_available = HAS_SHAP
        if not self.shap_available:
            self.logger.warning("SHAP package not available. SHAP visualizations will be unavailable.")
        
        self.logger.info("FeatureVisualizer initialized")
    
    def plot_feature_importance(self, importance_data, title=None, top_n=20, 
                               figsize=(12, 10), output_file=None, error_bars=True):
        """
        Create a bar chart of feature importance.
        
        Args:
            importance_data (dict): Feature importance data
            title (str, optional): Plot title
            top_n (int, optional): Number of top features to show
            figsize (tuple, optional): Figure size
            output_file (str, optional): Path to save the plot
            error_bars (bool): Whether to include error bars if available
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info(f"Plotting feature importance (top {top_n} features)")
        
        try:
            # Extract feature rankings
            if 'mean_importance' in importance_data:
                # Permutation importance or SHAP values format
                importance_values = importance_data['mean_importance']
                std_values = importance_data.get('std_importance', {})
            elif isinstance(importance_data, dict) and all(isinstance(v, (int, float)) for v in importance_data.values()):
                # Simple {feature: importance} format
                importance_values = importance_data
                std_values = {}
            else:
                self.logger.error("Unrecognized importance data format")
                return None, None
            
            # Sort features by importance
            sorted_features = sorted(importance_values.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to top_n features
            sorted_features = sorted_features[:top_n]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Extract feature names and values
            features, values = zip(*sorted_features)
            
            # Prepare X positions and flip for horizontal bars
            y_pos = range(len(features))
            
            # Create horizontal bar chart
            bars = ax.barh(y_pos, values, align='center')
            
            # Add error bars if available and requested
            if error_bars and std_values:
                errors = [std_values.get(feature, 0) for feature in features]
                ax.errorbar(values, y_pos, xerr=errors, fmt='none', ecolor='black', capsize=5)
            
            # Add feature names to Y axis
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            
            # Invert y axis to show most important at the top
            ax.invert_yaxis()
            
            # Add title and labels
            if title:
                ax.set_title(title, fontsize=14)
            else:
                method_name = importance_data.get('method', 'Feature').replace('_', ' ').title()
                ax.set_title(f"{method_name} Importance", fontsize=14)
            
            ax.set_xlabel('Importance')
            
            # Add grid lines for horizontal bars
            ax.grid(axis='x')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved importance plot to {output_file}")
            
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            return None, None
    
    def plot_importance_comparison(self, importance_data_list, method_names=None, top_n=15,
                                  figsize=(14, 10), output_file=None):
        """
        Create a grouped bar chart comparing feature importance from multiple methods.
        
        Args:
            importance_data_list (list): List of feature importance data dictionaries
            method_names (list, optional): Names for each importance method
            top_n (int, optional): Number of top features to show
            figsize (tuple, optional): Figure size
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info(f"Plotting feature importance comparison across {len(importance_data_list)} methods")
        
        try:
            # Set method names if not provided
            if method_names is None:
                method_names = []
                for i, data in enumerate(importance_data_list):
                    if 'method' in data:
                        method_names.append(data['method'].replace('_', ' ').title())
                    else:
                        method_names.append(f"Method {i+1}")
            
            # Extract feature rankings from each method
            all_rankings = []
            
            for data in importance_data_list:
                if 'mean_importance' in data:
                    # Permutation importance or SHAP values format
                    importance_values = data['mean_importance']
                elif isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
                    # Simple {feature: importance} format
                    importance_values = data
                else:
                    self.logger.warning("Skipping unrecognized importance data format")
                    continue
                
                # Sort features by importance
                sorted_features = sorted(importance_values.items(), key=lambda x: x[1], reverse=True)
                all_rankings.append(sorted_features)
            
            if not all_rankings:
                self.logger.error("No valid feature rankings to compare")
                return None, None
            
            # Collect all features from top_n of each method
            all_features = set()
            for rankings in all_rankings:
                all_features.update(f for f, _ in rankings[:top_n])
            
            # Create a combined importance dictionary
            combined_importance = {}
            
            for feature in all_features:
                # Get importance from each method
                feature_importance = []
                for rankings in all_rankings:
                    # Find importance in this ranking
                    importance = 0
                    for f, v in rankings:
                        if f == feature:
                            importance = v
                            break
                    feature_importance.append(importance)
                
                # Calculate average importance
                combined_importance[feature] = np.mean(feature_importance)
            
            # Sort by combined importance
            sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to top_n features
            top_features = [f for f, _ in sorted_features[:top_n]]
            
            # Prepare data for plotting
            plot_data = []
            
            for i, rankings in enumerate(all_rankings):
                importance_dict = dict(rankings)
                for feature in top_features:
                    plot_data.append({
                        'Feature': feature,
                        'Importance': importance_dict.get(feature, 0),
                        'Method': method_names[i]
                    })
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(plot_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create grouped bar chart
            sns.barplot(x='Feature', y='Importance', hue='Method', data=df, ax=ax)
            
            # Adjust layout
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add title and labels
            ax.set_title("Feature Importance Comparison", fontsize=14)
            ax.set_xlabel('')
            ax.set_ylabel('Importance')
            
            # Add legend with title
            ax.legend(title='Method')
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved comparison plot to {output_file}")
            
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"Error plotting importance comparison: {str(e)}")
            return None, None
    
    def plot_correlation_heatmap(self, X, feature_names=None, figsize=(12, 10), output_file=None):
        """
        Create a correlation heatmap of features.
        
        Args:
            X (numpy.ndarray or pandas.DataFrame): Feature matrix
            feature_names (list, optional): Names of features
            figsize (tuple, optional): Figure size
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Plotting feature correlation heatmap")
        
        try:
            # Convert to DataFrame if not already
            if isinstance(X, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=feature_names)
            else:
                df = X.copy()
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                       square=True, linewidths=.5, annot=False, cbar_kws={"shrink": .5}, ax=ax)
            
            # Add title
            ax.set_title('Feature Correlation Heatmap', fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved correlation heatmap to {output_file}")
            
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"Error plotting correlation heatmap: {str(e)}")
            return None, None
    
    def plot_shap_summary(self, shap_data, X_sample=None, max_display=20, output_file=None):
        """
        Create a SHAP summary plot.
        
        Args:
            shap_data (dict): SHAP value data from FeatureAnalyzer
            X_sample (numpy.ndarray, optional): Sample data for plot
            max_display (int, optional): Maximum number of features to display
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        if not self.shap_available:
            self.logger.error("SHAP package not available. Cannot create SHAP summary plot.")
            return None, None
        
        self.logger.info("Plotting SHAP summary")
        
        try:
            # Check if we have required SHAP data
            if '_values' not in shap_data:
                self.logger.error("SHAP values not found in data")
                return None, None
            
            # Use provided sample data or stored sample
            if X_sample is None and '_sample' in shap_data:
                X_sample = shap_data['_sample']
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create summary plot
            shap_values = shap_data['_values']
            
            # Handle different formats of SHAP values
            if isinstance(shap_values, list):
                # Multi-class or multi-output, use first class
                shap.summary_plot(shap_values[0], X_sample, max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
            
            # Get current figure and axes
            fig = plt.gcf()
            ax = plt.gca()
            
            # Add title
            plt.title("SHAP Feature Importance", fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved SHAP summary plot to {output_file}")
            
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP summary: {str(e)}")
            return None, None
    
    def plot_shap_dependence(self, shap_data, feature, interact_feature=None, output_file=None):
        """
        Create a SHAP dependence plot for a feature.
        
        Args:
            shap_data (dict): SHAP value data from FeatureAnalyzer
            feature (str or int): Feature to analyze
            interact_feature (str or int, optional): Second feature for interaction
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        if not self.shap_available:
            self.logger.error("SHAP package not available. Cannot create SHAP dependence plot.")
            return None, None
        
        self.logger.info(f"Plotting SHAP dependence for feature {feature}")
        
        try:
            # Check if we have required SHAP data
            if '_values' not in shap_data or '_sample' not in shap_data:
                self.logger.error("SHAP values or sample data not found")
                return None, None
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create dependence plot
            shap_values = shap_data['_values']
            X_sample = shap_data['_sample']
            
            # Handle different formats of SHAP values
            if isinstance(shap_values, list):
                # Multi-class or multi-output, use first class
                shap.dependence_plot(
                    feature, shap_values[0], X_sample, 
                    interaction_index=interact_feature, 
                    show=False
                )
            else:
                shap.dependence_plot(
                    feature, shap_values, X_sample, 
                    interaction_index=interact_feature, 
                    show=False
                )
            
            # Get current figure and axes
            fig = plt.gcf()
            ax = plt.gca()
            
            # Add title
            plt.title(f"SHAP Dependence Plot for {feature}", fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved SHAP dependence plot to {output_file}")
            
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"Error plotting SHAP dependence: {str(e)}")
            return None, None
    
    def plot_partial_dependence(self, model, X, feature_names, top_n=6, figsize=(14, 10), output_file=None):
        """
        Create partial dependence plots for top features.
        
        Args:
            model: Trained model
            X (numpy.ndarray): Feature matrix
            feature_names (list): Names of features
            top_n (int, optional): Number of top features to plot
            figsize (tuple, optional): Figure size
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info(f"Plotting partial dependence for top {top_n} features")
        
        try:
            # Determine top features using built-in importance if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                top_indices = np.argsort(importance)[-top_n:]
                
                # Reverse for descending order
                top_indices = top_indices[::-1]
                
                # Get feature names for top features
                top_features = [feature_names[i] for i in top_indices]
            else:
                # Use the first top_n features if importance not available
                top_features = feature_names[:min(top_n, len(feature_names))]
                top_indices = list(range(min(top_n, len(feature_names))))
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create partial dependence plot
            pdp = plot_partial_dependence(
                model, X, features=top_indices, 
                feature_names=feature_names,
                n_jobs=-1, grid_resolution=20,
                n_cols=3, ax=ax  # Arrange in 3 columns
            )
            
            # Add title
            fig.suptitle("Partial Dependence Plots for Top Features", fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved partial dependence plot to {output_file}")
            
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"Error plotting partial dependence: {str(e)}")
            return None, None
    
    def plot_time_based_importance(self, time_importance_data, top_n=10, figsize=(14, 8), output_file=None):
        """
        Visualize how feature importance changes over time.
        
        Args:
            time_importance_data (dict): Time-based feature importance data
            top_n (int, optional): Number of top features to display
            figsize (tuple, optional): Figure size
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Plotting time-based feature importance")
        
        try:
            # Check if we have period importance data
            if 'period_importance' not in time_importance_data:
                self.logger.error("Period importance data not found")
                return None, None
            
            # Extract period importance and labels
            period_importance = time_importance_data['period_importance']
            period_labels = time_importance_data.get('time_periods', list(period_importance.keys()))
            
            # Get feature trends
            feature_trends = time_importance_data.get('feature_trends', {})
            
            # Determine top features by change in importance
            if 'features_by_change' in time_importance_data:
                top_features = time_importance_data['features_by_change'][:top_n]
            else:
                # Sort features by mean importance
                top_features = sorted(
                    [(f, feature_trends[f]['mean']) for f in feature_trends],
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                top_features = [f[0] for f in top_features]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Create line plot for feature importance over time
            for feature in top_features:
                if feature in feature_trends:
                    values = feature_trends[feature]['values']
                    ax1.plot(period_labels, values, marker='o', linewidth=2, label=feature)
            
            # Configure line plot
            ax1.set_title('Feature Importance Over Time', fontsize=14)
            ax1.set_xlabel('Time Period')
            ax1.set_ylabel('Importance')
            ax1.legend(loc='best')
            ax1.grid(True)
            
            # Create heatmap for top features across periods
            heatmap_data = []
            
            for feature in top_features:
                feature_data = []
                for period in period_labels:
                    if period in period_importance:
                        feature_data.append(period_importance[period].get(feature, 0))
                    else:
                        feature_data.append(0)
                heatmap_data.append(feature_data)
            
            # Create heatmap
            im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax2)
            cbar.set_label('Importance')
            
            # Configure heatmap
            ax2.set_title('Feature Importance Heatmap', fontsize=14)
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features)
            ax2.set_xticks(range(len(period_labels)))
            ax2.set_xticklabels(period_labels, rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved time-based importance plot to {output_file}")
            
            return fig, ax1, ax2
            
        except Exception as e:
            self.logger.error(f"Error plotting time-based importance: {str(e)}")
            return None, None, None
    
    def plot_feature_stability(self, importance_data, top_n=20, figsize=(12, 8), output_file=None):
        """
        Create a plot showing feature importance stability.
        
        Args:
            importance_data (dict): Feature importance data with standard deviations
            top_n (int, optional): Number of top features to display
            figsize (tuple, optional): Figure size
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info(f"Plotting feature importance stability for top {top_n} features")
        
        try:
            # Check if we have standard deviation data
            if 'std_importance' not in importance_data or 'mean_importance' not in importance_data:
                self.logger.error("Standard deviation data not found")
                return None, None
            
            # Extract mean and std importance
            mean_importance = importance_data['mean_importance']
            std_importance = importance_data['std_importance']
            
            # Sort features by mean importance
            sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top N features
            top_features = sorted_features[:top_n]
            
            # Calculate coefficient of variation (CV)
            cv_values = {}
            for feature, mean in top_features:
                std = std_importance.get(feature, 0)
                cv = std / mean if mean > 0 else 0
                cv_values[feature] = cv
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Extract feature names and values
            features, means = zip(*top_features)
            stds = [std_importance.get(feature, 0) for feature in features]
            cvs = [cv_values[feature] for feature in features]
            
            # Create bar chart for mean importance
            x = np.arange(len(features))
            width = 0.35
            bars = ax.bar(x, means, width, yerr=stds, alpha=0.7, 
                        capsize=5, label='Mean Importance')
            
            # Create second y-axis for coefficient of variation
            ax2 = ax.twinx()
            ax2.plot(x, cvs, 'ro-', linewidth=2, label='Coefficient of Variation')
            
            # Configure axes
            ax.set_xlabel('Features')
            ax.set_ylabel('Mean Importance')
            ax2.set_ylabel('Coefficient of Variation')
            
            # Set ticks
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45, ha='right')
            
            # Add title
            plt.title('Feature Importance Stability', fontsize=14)
            
            # Add legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved stability plot to {output_file}")
            
            return fig, (ax, ax2)
            
        except Exception as e:
            self.logger.error(f"Error plotting feature stability: {str(e)}")
            return None, None
    
    def plot_categorical_feature_analysis(self, cat_feature_data, top_n=5, figsize=(14, 10), output_file=None):
        """
        Visualize categorical feature analysis results.
        
        Args:
            cat_feature_data (dict): Categorical feature analysis data
            top_n (int, optional): Number of top features to display
            figsize (tuple, optional): Figure size
            output_file (str, optional): Path to save the plot
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Plotting categorical feature analysis")
        
        try:
            # Check if we have categorical feature data
            if 'categorical_features' not in cat_feature_data:
                self.logger.error("Categorical feature data not found")
                return None, None
            
            # Extract categorical features and importance scores
            categorical_features = cat_feature_data['categorical_features']
            
            # Get top features by importance score
            if 'sorted_by_importance' in cat_feature_data:
                top_features = cat_feature_data['sorted_by_importance'][:top_n]
            else:
                top_features = sorted(
                    [(f, categorical_features[f]['importance_score']) for f in categorical_features],
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                top_features = [f[0] for f in top_features]
            
            # Create figure with subplots for each top feature
            n_features = len(top_features)
            n_cols = min(2, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            
            # Flatten axes array for easier indexing
            if n_features > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            # Plot each feature
            for i, feature in enumerate(top_features):
                if i >= len(axes):
                    break
                    
                if feature not in categorical_features:
                    continue
                
                # Get data for this feature
                feature_data = categorical_features[feature]
                means = feature_data['category_means']
                counts = feature_data['category_counts']
                
                # Sort categories by mean target value
                sorted_cats = sorted(means.items(), key=lambda x: x[1])
                cats, cat_means = zip(*sorted_cats)
                
                # Get counts for these categories
                cat_counts = [counts.get(cat, 0) for cat in cats]
                
                # Create bar chart
                ax = axes[i]
                bars = ax.bar(range(len(cats)), cat_means, alpha=0.7)
                
                # Color bars based on mean (green for high, red for low)
                for j, bar in enumerate(bars):
                    normalized_mean = (cat_means[j] - min(cat_means)) / (max(cat_means) - min(cat_means)) if max(cat_means) > min(cat_means) else 0.5
                    bar.set_color(plt.cm.RdYlGn(normalized_mean))
                
                # Add count as text on bars
                for j, count in enumerate(cat_counts):
                    ax.text(j, cat_means[j], f"n={count}", ha='center', va='bottom')
                
                # Configure axis
                ax.set_title(feature, fontsize=12)
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels(cats, rotation=45, ha='right')
                ax.set_ylabel('Mean Target Value')
                
                # Add horizontal line for global mean
                global_mean = sum(m * counts[c] for c, m in means.items()) / sum(counts.values()) if sum(counts.values()) > 0 else 0
                ax.axhline(y=global_mean, color='r', linestyle='--', alpha=0.5)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            # Add overall title
            fig.suptitle('Categorical Feature Analysis', fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved categorical feature analysis plot to {output_file}")
            
            return fig, axes
            
        except Exception as e:
            self.logger.error(f"Error plotting categorical feature analysis: {str(e)}")
            return None, None
    
    def create_feature_importance_dashboard(self, model, X, y, feature_names, output_file=None):
        """
        Create a comprehensive dashboard of feature importance visualizations.
        
        Args:
            model: Trained model
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list): Names of features
            output_file (str, optional): Path to save the dashboard
            
        Returns:
            tuple: Figure and axes
        """
        self.logger.info("Creating feature importance dashboard")
        
        try:
            # Create a large figure
            fig = plt.figure(figsize=(20, 16))
            
            # Define grid layout
            gs = fig.add_gridspec(3, 3)
            
            # 1. Built-in feature importance (if available)
            ax1 = fig.add_subplot(gs[0, 0])
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                sorted_idx = np.argsort(importance)[-15:]  # Top 15 features
                y_pos = np.arange(len(sorted_idx))
                ax1.barh(y_pos, importance[sorted_idx])
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
                ax1.set_title('Built-in Feature Importance')
                ax1.invert_yaxis()
            else:
                ax1.text(0.5, 0.5, "Built-in importance not available", 
                       ha='center', va='center', transform=ax1.transAxes)
            
            # 2. Feature correlation heatmap
            ax2 = fig.add_subplot(gs[0, 1:])
            corr_matrix = np.corrcoef(X.T)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                       xticklabels=feature_names, yticklabels=feature_names, ax=ax2)
            ax2.set_title('Feature Correlation Heatmap')
            
            # 3. Partial dependence plots
            ax3 = fig.add_subplot(gs[1, :])
            
            try:
                # Get top features
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    top_indices = np.argsort(importance)[-3:]  # Top 3 features
                    # Create partial dependence plot
                    plot_partial_dependence(
                        model, X, features=top_indices, 
                        feature_names=feature_names,
                        n_cols=3, ax=ax3
                    )
                    ax3.get_figure().suptitle('')  # Remove default suptitle
                    ax3.set_title('Partial Dependence Plots')
                else:
                    ax3.text(0.5, 0.5, "Partial dependence not available", 
                           ha='center', va='center', transform=ax3.transAxes)
            except Exception as pdp_error:
                self.logger.warning(f"Error creating partial dependence plot: {str(pdp_error)}")
                ax3.text(0.5, 0.5, "Error creating partial dependence plot", 
                       ha='center', va='center', transform=ax3.transAxes)
            
            # 4. SHAP summary plot (if available)
            ax4 = fig.add_subplot(gs[2, :])
            
            if self.shap_available:
                try:
                    # Sample data for SHAP analysis
                    sample_size = min(100, X.shape[0])
                    sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
                    X_sample = X[sample_indices]
                    
                    # Create SHAP explainer
                    explainer = shap.Explainer(model, X_sample)
                    shap_values = explainer(X_sample)
                    
                    # Create summary plot
                    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                     max_display=10, show=False, plot_size=[0.4, 0.4],
                                     plot_type='bar', ax=ax4)
                    ax4.set_title('SHAP Feature Importance')
                except Exception as shap_error:
                    self.logger.warning(f"Error creating SHAP plot: {str(shap_error)}")
                    ax4.text(0.5, 0.5, "Error creating SHAP plot", 
                           ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, "SHAP package not available", 
                       ha='center', va='center', transform=ax4.transAxes)
            
            # Add dashboard title
            fig.suptitle('Feature Importance Dashboard', fontsize=20)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save figure if output file specified
            if output_file:
                fig.savefig(output_file, bbox_inches='tight')
                self.logger.info(f"Saved feature importance dashboard to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance dashboard: {str(e)}")
            return None
    
    def generate_feature_report_pdf(self, importance_data, model_name, output_file=None):
        """
        Generate a PDF report with feature importance visualizations.
        
        Args:
            importance_data (dict): Feature importance data
            model_name (str): Name of the model
            output_file (str, optional): Path to save the PDF report
            
        Returns:
            bool: True if successful
        """
        try:
            # Try to import libraries for PDF creation
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib import colors
                from io import BytesIO
                REPORTLAB_AVAILABLE = True
            except ImportError:
                self.logger.warning("ReportLab not available. Cannot create PDF report.")
                return False
                
            # Set default output file if not provided
            if output_file is None:
                output_file = os.path.join(self.plots_dir, f"{model_name}_feature_report.pdf")
            
            # Create document
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Create report content
            content = []
            
            # Add title
            title_style = styles["Title"]
            content.append(Paragraph(f"Feature Importance Report: {model_name}", title_style))
            content.append(Spacer(1, 12))
            
            # Add date
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            content.append(Spacer(1, 12))
            
            # Add method information
            if 'method' in importance_data:
                method_name = importance_data['method'].replace('_', ' ').title()
                content.append(Paragraph(f"Method: {method_name}", styles["Heading2"]))
                content.append(Spacer(1, 12))
            
            # Create feature importance plot
            fig, ax = self.plot_feature_importance(importance_data, top_n=15)
            if fig:
                # Save plot to bytesIO
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                
                # Add plot to report
                img = Image(img_buffer, width=450, height=350)
                content.append(img)
                content.append(Spacer(1, 12))
                
                # Close figure to free memory
                plt.close(fig)
            
            # Add feature importance table
            content.append(Paragraph("Feature Importance Ranking", styles["Heading2"]))
            content.append(Spacer(1, 12))
            
            # Get feature ranking
            if 'mean_importance' in importance_data:
                feature_ranking = sorted(importance_data['mean_importance'].items(), key=lambda x: x[1], reverse=True)
            elif isinstance(importance_data, dict) and all(isinstance(v, (int, float)) for v in importance_data.values()):
                feature_ranking = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
            else:
                feature_ranking = []
            
            # Create table data
            table_data = [["Rank", "Feature", "Importance"]]
            for i, (feature, importance) in enumerate(feature_ranking[:20]):  # Top 20 features
                table_data.append([str(i+1), feature, f"{importance:.6f}"])
            
            # Create table
            if len(table_data) > 1:
                table = Table(table_data, colWidths=[40, 300, 100])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                content.append(table)
                content.append(Spacer(1, 24))
            
            # Add method-specific information
            if 'method' in importance_data:
                method = importance_data['method']
                
                if method == 'permutation_importance':
                    content.append(Paragraph("Permutation Importance Details", styles["Heading2"]))
                    content.append(Spacer(1, 12))
                    content.append(Paragraph(f"Number of permutation repeats: {importance_data.get('n_repeats', 'N/A')}", styles["Normal"]))
                    content.append(Spacer(1, 12))
                    
                    # Create stability plot if std_importance is available
                    if 'std_importance' in importance_data:
                        fig, ax = self.plot_feature_stability(importance_data)
                        if fig:
                            # Save plot to bytesIO
                            img_buffer = BytesIO()
                            fig.savefig(img_buffer, format='png', bbox_inches='tight')
                            img_buffer.seek(0)
                            
                            # Add plot to report
                            img = Image(img_buffer, width=450, height=350)
                            content.append(img)
                            
                            # Close figure to free memory
                            plt.close(fig)
                
                elif method == 'shap_values':
                    content.append(Paragraph("SHAP Value Details", styles["Heading2"]))
                    content.append(Spacer(1, 12))
                    content.append(Paragraph(f"Sample size used: {importance_data.get('sample_size', 'N/A')}", styles["Normal"]))
                    content.append(Paragraph("SHAP values represent the contribution of each feature to model predictions", styles["Normal"]))
            
            # Build document
            doc.build(content)
            
            self.logger.info(f"Generated feature report PDF: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating feature report PDF: {str(e)}")
            return False
