"""
Feature Analyzer Module

This module provides tools for analyzing feature importance in machine learning models,
including permutation importance, SHAP values, and partial dependence plots.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.inspection import permutation_importance, partial_dependence, plot_partial_dependence
from sklearn.preprocessing import LabelEncoder

# Try to import SHAP - an optional dependency
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class FeatureAnalyzer:
    """
    Provides methods for analyzing feature importance in machine learning models.
    
    Features:
    - Built-in feature importance extraction for tree-based models
    - Permutation importance calculation
    - SHAP value analysis for complex feature interactions
    - Partial dependence analysis for feature effects
    """
    
    def __init__(self, config=None):
        """
        Initialize the FeatureAnalyzer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.results_dir = os.path.join("data", "feature_importance")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default parameters
        self.n_permutations = self.config.get("n_permutations", 10)
        self.random_state = self.config.get("random_state", 42)
        self.n_jobs = self.config.get("n_jobs", -1)
        
        # Flag to indicate if SHAP is available
        self.shap_available = HAS_SHAP
        if not self.shap_available:
            self.logger.warning("SHAP package not available. SHAP-based methods will be unavailable.")
        
        self.logger.info("FeatureAnalyzer initialized")
    
    def extract_feature_importance(self, model, feature_names=None):
        """
        Extract built-in feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (list, optional): Names of features
            
        Returns:
            dict: Feature importance values mapped to feature names
        """
        self.logger.info("Extracting built-in feature importance")
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning("Model does not have built-in feature_importances_ attribute")
            return None
        
        # Get feature importance values
        importance_values = model.feature_importances_
        
        # If feature names not provided, use generic names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
        
        # Create dictionary mapping feature names to importance values
        importance_dict = dict(zip(feature_names, importance_values))
        
        # Sort by importance (descending)
        importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
        
        self.logger.info(f"Extracted {len(importance_dict)} feature importance values")
        
        return importance_dict
    
    def calculate_permutation_importance(self, model, X, y, feature_names=None, n_repeats=None):
        """
        Calculate permutation-based feature importance.
        
        Args:
            model: Trained model
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list, optional): Names of features
            n_repeats (int, optional): Number of permutation repeats
            
        Returns:
            dict: Permutation importance results
        """
        self.logger.info("Calculating permutation importance")
        
        # Use default n_repeats if not specified
        if n_repeats is None:
            n_repeats = self.n_permutations
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X, y, 
            n_repeats=n_repeats, 
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # If feature names not provided, use generic names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create result dictionary
        importance_dict = {
            'mean_importance': dict(zip(feature_names, result.importances_mean)),
            'std_importance': dict(zip(feature_names, result.importances_std)),
            'raw_importances': dict(zip(feature_names, [imp.tolist() for imp in result.importances])),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'permutation_importance',
            'n_repeats': n_repeats
        }
        
        # Sort mean importance by value (descending)
        importance_dict['mean_importance'] = {
            k: v for k, v in sorted(
                importance_dict['mean_importance'].items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        }
        
        self.logger.info(f"Calculated permutation importance for {len(feature_names)} features")
        
        return importance_dict
    
    def calculate_shap_values(self, model, X, feature_names=None, sample_size=None, model_output=None):
        """
        Calculate SHAP values for feature importance.
        
        Args:
            model: Trained model
            X (numpy.ndarray): Feature matrix
            feature_names (list, optional): Names of features
            sample_size (int, optional): Number of samples to use (for large datasets)
            model_output (str, optional): For multi-output models, which output to explain
            
        Returns:
            dict: SHAP importance results
        """
        if not self.shap_available:
            self.logger.error("SHAP package not available. Cannot calculate SHAP values.")
            return None
        
        self.logger.info("Calculating SHAP values")
        
        # If feature names not provided, use generic names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Use a sample of data if specified
        if sample_size is not None and sample_size < X.shape[0]:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
            X_sample = X[sample_indices]
        else:
            X_sample = X
        
        try:
            # Determine the appropriate SHAP explainer
            if hasattr(model, 'predict_proba'):
                # For classifiers that provide predict_proba
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
            elif hasattr(model, 'tree_'):
                # For single tree models (decision trees)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            elif hasattr(model, 'estimators_'):
                # For random forests and similar ensemble models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # For other models, use KernelExplainer
                explainer = shap.KernelExplainer(model.predict, X_sample[:50])  # Use subset for background
                shap_values = explainer.shap_values(X_sample)
            
            # Handle different shapes of SHAP values
            if isinstance(shap_values, list):
                # Multiple outputs (e.g., for classifier with >2 classes)
                if model_output is None:
                    # Default to first output for multi-output models
                    model_output = 0
                
                # Select the specified output
                if isinstance(model_output, int) and model_output < len(shap_values):
                    values = shap_values[model_output]
                else:
                    values = shap_values[0]  # Default to first output
            else:
                # Single output
                values = shap_values.values
            
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.mean(np.abs(values), axis=0)
            
            # Create result dictionary
            shap_dict = {
                'mean_importance': dict(zip(feature_names, mean_abs_shap.tolist())),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'shap_values',
                'sample_size': X_sample.shape[0]
            }
            
            # Sort mean importance by value (descending)
            shap_dict['mean_importance'] = {
                k: v for k, v in sorted(
                    shap_dict['mean_importance'].items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )
            }
            
            # Store the explainer and values for visualization
            shap_dict['_explainer'] = explainer
            shap_dict['_values'] = shap_values
            shap_dict['_sample'] = X_sample
            
            self.logger.info(f"Calculated SHAP values for {len(feature_names)} features")
            
            return shap_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {str(e)}")
            return None
    
    def calculate_partial_dependence(self, model, X, feature_names, n_features=5, grid_resolution=20):
        """
        Calculate partial dependence plots for top features.
        
        Args:
            model: Trained model
            X (numpy.ndarray): Feature matrix
            feature_names (list): Names of features
            n_features (int, optional): Number of top features to analyze
            grid_resolution (int, optional): Resolution of the grid for PDPs
            
        Returns:
            dict: Partial dependence results
        """
        self.logger.info(f"Calculating partial dependence for top {n_features} features")
        
        try:
            # Determine top features using built-in importance if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                top_indices = np.argsort(importance)[-n_features:]
            else:
                # Use the first n_features if importance not available
                top_indices = np.arange(min(n_features, len(feature_names)))
            
            # Reverse for descending order
            top_indices = top_indices[::-1]
            
            # Get feature names for top features
            top_features = [feature_names[i] for i in top_indices]
            
            # Calculate partial dependence
            pd_result = {}
            
            for feature_idx, feature_name in zip(top_indices, top_features):
                # Single feature PDP
                pd_values = partial_dependence(
                    model, X, features=[feature_idx], 
                    grid_resolution=grid_resolution,
                    method='brute'
                )
                
                # Store results
                pd_result[feature_name] = {
                    'values': pd_values['values'][0].tolist(),
                    'grid_points': pd_values['grid_points'][0].tolist()
                }
            
            # Add metadata
            pd_result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pd_result['method'] = 'partial_dependence'
            pd_result['grid_resolution'] = grid_resolution
            
            self.logger.info(f"Calculated partial dependence for {len(top_features)} features")
            
            return pd_result
            
        except Exception as e:
            self.logger.error(f"Error calculating partial dependence: {str(e)}")
            return None
    
    def analyze_feature_correlations(self, X, feature_names=None):
        """
        Analyze correlations between features.
        
        Args:
            X (numpy.ndarray or pandas.DataFrame): Feature matrix
            feature_names (list, optional): Names of features
            
        Returns:
            dict: Correlation analysis results
        """
        self.logger.info("Analyzing feature correlations")
        
        try:
            # Convert to DataFrame if not already
            if isinstance(X, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=feature_names)
            else:
                df = X.copy()
                feature_names = df.columns.tolist()
            
            # Calculate correlation matrix
            corr_matrix = df.corr().fillna(0)
            
            # Find highly correlated features
            high_corr_threshold = 0.7
            high_corr_pairs = []
            
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                        high_corr_pairs.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            # Sort pairs by absolute correlation (descending)
            high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
            
            # Create result dictionary
            result = {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlation_pairs': high_corr_pairs,
                'high_correlation_threshold': high_corr_threshold,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'correlation_analysis'
            }
            
            self.logger.info(f"Analyzed correlations for {len(feature_names)} features")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature correlations: {str(e)}")
            return None
    
    def analyze_categorical_features(self, X, y, feature_names=None):
        """
        Analyze the importance of categorical features.
        
        Args:
            X (numpy.ndarray or pandas.DataFrame): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list, optional): Names of features
            
        Returns:
            dict: Categorical feature analysis results
        """
        self.logger.info("Analyzing categorical features")
        
        try:
            # Convert to DataFrame if not already
            if isinstance(X, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=feature_names)
            else:
                df = X.copy()
                feature_names = df.columns.tolist()
            
            # Add target to the dataframe
            df['target'] = y
            
            # Identify categorical features
            categorical_features = []
            for col in feature_names:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    categorical_features.append(col)
            
            if not categorical_features:
                self.logger.info("No categorical features found")
                return None
            
            # Analyze each categorical feature
            results = {}
            
            for feature in categorical_features:
                # Calculate target mean for each category
                category_means = df.groupby(feature)['target'].mean().to_dict()
                
                # Calculate target count for each category
                category_counts = df.groupby(feature)['target'].count().to_dict()
                
                # Find the most predictive values (highest/lowest target means)
                sorted_means = sorted(category_means.items(), key=lambda x: x[1])
                lowest_means = sorted_means[:3]
                highest_means = sorted_means[-3:]
                
                # Store results
                results[feature] = {
                    'category_means': category_means,
                    'category_counts': category_counts,
                    'lowest_mean_categories': lowest_means,
                    'highest_mean_categories': highest_means,
                    'n_categories': len(category_means),
                    'importance_score': np.std(list(category_means.values())) if len(category_means) > 1 else 0
                }
            
            # Sort features by importance score
            sorted_features = sorted(
                [(f, results[f]['importance_score']) for f in categorical_features],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create final result
            analysis_result = {
                'categorical_features': results,
                'sorted_by_importance': [f[0] for f in sorted_features],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'categorical_feature_analysis'
            }
            
            self.logger.info(f"Analyzed {len(categorical_features)} categorical features")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing categorical features: {str(e)}")
            return None
    
    def analyze_time_based_importance(self, model, X, feature_names, time_periods, period_labels=None):
        """
        Analyze how feature importance changes over different time periods.
        
        Args:
            model: Trained model with feature_importances_ attribute
            X (numpy.ndarray): Feature matrix with samples from different time periods
            feature_names (list): Names of features
            time_periods (list): List of indices indicating which time period each sample belongs to
            period_labels (list, optional): Names for the time periods
            
        Returns:
            dict: Time-based feature importance analysis
        """
        self.logger.info("Analyzing time-based feature importance")
        
        try:
            # Verify that model has feature importances
            if not hasattr(model, 'feature_importances_'):
                self.logger.warning("Model does not have built-in feature_importances_ attribute")
                return None
            
            # If period labels not provided, use generic names
            if period_labels is None:
                unique_periods = sorted(np.unique(time_periods))
                period_labels = [f"period_{p}" for p in unique_periods]
            else:
                unique_periods = list(range(len(period_labels)))
            
            # Analyze feature importance for each time period
            period_importance = {}
            
            for period_idx, period_label in zip(unique_periods, period_labels):
                # Extract samples for this time period
                period_mask = np.array(time_periods) == period_idx
                X_period = X[period_mask]
                
                if X_period.shape[0] == 0:
                    self.logger.warning(f"No samples found for period {period_label}")
                    continue
                
                # Determine feature importance method
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models, retrain on this period's data
                    import copy
                    model_copy = copy.deepcopy(model)
                    model_copy.fit(X_period, y[period_mask])
                    importance_values = model_copy.feature_importances_
                else:
                    # Use permutation importance for other models
                    result = permutation_importance(
                        model, X_period, y[period_mask], 
                        n_repeats=5, 
                        random_state=self.random_state,
                        n_jobs=self.n_jobs
                    )
                    importance_values = result.importances_mean
                
                # Map to feature names
                period_importance[period_label] = dict(zip(feature_names, importance_values))
            
            # Find features with significant changes in importance
            feature_trends = {}
            
            for feature in feature_names:
                importance_values = [period_importance[period].get(feature, 0) for period in period_labels if period in period_importance]
                
                if len(importance_values) < 2:
                    continue
                
                # Calculate trend metrics
                feature_trends[feature] = {
                    'mean': np.mean(importance_values),
                    'std': np.std(importance_values),
                    'min': np.min(importance_values),
                    'max': np.max(importance_values),
                    'change': importance_values[-1] - importance_values[0],
                    'values': importance_values
                }
            
            # Sort features by absolute change
            sorted_features = sorted(
                [(f, abs(feature_trends[f]['change'])) for f in feature_trends],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create final result
            analysis_result = {
                'period_importance': period_importance,
                'feature_trends': feature_trends,
                'features_by_change': [f[0] for f in sorted_features],
                'time_periods': period_labels,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'time_based_importance_analysis'
            }
            
            self.logger.info(f"Analyzed time-based importance for {len(feature_names)} features across {len(period_labels)} periods")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing time-based importance: {str(e)}")
            return None
    
    def save_feature_importance(self, importance_data, model_name, method="built_in"):
        """
        Save feature importance results to a file.
        
        Args:
            importance_data (dict): Feature importance data
            model_name (str): Name of the model
            method (str): Method used to calculate importance
            
        Returns:
            str: Path to the saved file
        """
        # Create a safe filename
        safe_name = model_name.replace(' ', '_').lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_name}_{method}_importance_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare data for saving
        save_data = importance_data.copy()
        
        # Remove non-serializable objects
        for key in list(save_data.keys()):
            if key.startswith('_'):
                del save_data[key]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Saved feature importance results to {filepath}")
        
        return filepath
    
    def load_feature_importance(self, filepath):
        """
        Load feature importance results from a file.
        
        Args:
            filepath (str): Path to the importance file
            
        Returns:
            dict: Feature importance data
        """
        try:
            with open(filepath, 'r') as f:
                importance_data = json.load(f)
            
            self.logger.info(f"Loaded feature importance results from {filepath}")
            return importance_data
        except Exception as e:
            self.logger.error(f"Error loading feature importance: {str(e)}")
            return None
    
    def get_feature_ranking(self, importance_data):
        """
        Get a ranked list of features from importance data.
        
        Args:
            importance_data (dict): Feature importance data
            
        Returns:
            list: Ranked list of (feature, importance) tuples
        """
        # Handle different importance data formats
        if 'mean_importance' in importance_data:
            # Permutation importance or SHAP values format
            importances = importance_data['mean_importance']
        elif isinstance(importance_data, dict) and all(isinstance(v, (int, float)) for v in importance_data.values()):
            # Simple {feature: importance} format
            importances = importance_data
        else:
            self.logger.error("Unrecognized importance data format")
            return []
        
        # Sort features by importance
        ranked_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_features
    
    def combine_importance_methods(self, importance_results, normalize=True):
        """
        Combine importance scores from multiple methods.
        
        Args:
            importance_results (list): List of importance data dictionaries
            normalize (bool): Whether to normalize importance values before combining
            
        Returns:
            dict: Combined feature importance scores
        """
        self.logger.info(f"Combining {len(importance_results)} feature importance methods")
        
        try:
            # Extract feature rankings from each method
            all_rankings = []
            
            for result in importance_results:
                if result is None:
                    continue
                    
                rankings = self.get_feature_ranking(result)
                if rankings:
                    all_rankings.append(rankings)
            
            if not all_rankings:
                self.logger.warning("No valid feature rankings to combine")
                return {}
            
            # Collect all unique features
            all_features = set()
            for rankings in all_rankings:
                all_features.update(f for f, _ in rankings)
            
            # Convert rankings to dictionaries
            ranking_dicts = []
            
            for rankings in all_rankings:
                ranking_dict = dict(rankings)
                
                # Add missing features with zero importance
                for feature in all_features:
                    if feature not in ranking_dict:
                        ranking_dict[feature] = 0.0
                
                # Normalize if requested
                if normalize:
                    max_value = max(ranking_dict.values())
                    if max_value > 0:
                        ranking_dict = {f: v / max_value for f, v in ranking_dict.items()}
                
                ranking_dicts.append(ranking_dict)
            
            # Combine importance scores
            combined_scores = {}
            
            for feature in all_features:
                # Take average importance across methods
                scores = [d[feature] for d in ranking_dicts]
                combined_scores[feature] = np.mean(scores)
            
            # Sort by combined importance
            combined_scores = {k: v for k, v in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)}
            
            self.logger.info(f"Combined importance scores for {len(combined_scores)} features")
            
            return combined_scores
            
        except Exception as e:
            self.logger.error(f"Error combining importance methods: {str(e)}")
            return {}
    
    def identify_redundant_features(self, importance_data, correlation_data, threshold=0.7):
        """
        Identify potentially redundant features based on importance and correlation.
        
        Args:
            importance_data (dict): Feature importance data
            correlation_data (dict): Feature correlation data
            threshold (float): Correlation threshold for redundancy
            
        Returns:
            list: List of potentially redundant features
        """
        self.logger.info(f"Identifying redundant features with threshold {threshold}")
        
        try:
            # Get feature ranking
            feature_ranking = self.get_feature_ranking(importance_data)
            
            # Extract high correlation pairs
            if 'high_correlation_pairs' in correlation_data:
                corr_pairs = correlation_data['high_correlation_pairs']
            else:
                corr_matrix = pd.DataFrame(correlation_data['correlation_matrix'])
                corr_pairs = []
                
                for i, f1 in enumerate(corr_matrix.columns):
                    for j, f2 in enumerate(corr_matrix.columns):
                        if i < j and abs(corr_matrix.iloc[i, j]) >= threshold:
                            corr_pairs.append({
                                'feature1': f1,
                                'feature2': f2,
                                'correlation': corr_matrix.iloc[i, j]
                            })
            
            # Map features to their importance
            feature_importance = dict(feature_ranking)
            
            # Identify redundant features
            redundant_features = []
            
            for pair in corr_pairs:
                f1, f2 = pair['feature1'], pair['feature2']
                
                if f1 in feature_importance and f2 in feature_importance:
                    # Determine which feature is less important
                    if feature_importance[f1] < feature_importance[f2]:
                        less_important = f1
                        more_important = f2
                    else:
                        less_important = f2
                        more_important = f1
                    
                    redundant_features.append({
                        'redundant_feature': less_important,
                        'keep_feature': more_important,
                        'importance_ratio': feature_importance[less_important] / feature_importance[more_important] if feature_importance[more_important] > 0 else 0,
                        'correlation': pair['correlation']
                    })
            
            # Sort by importance ratio (lower means more redundant)
            redundant_features = sorted(redundant_features, key=lambda x: x['importance_ratio'])
            
            self.logger.info(f"Identified {len(redundant_features)} potentially redundant features")
            
            return redundant_features
            
        except Exception as e:
            self.logger.error(f"Error identifying redundant features: {str(e)}")
            return []
    
    def generate_feature_importance_report(self, importance_data, model_name, output_file=None):
        """
        Generate a human-readable report of feature importance findings.
        
        Args:
            importance_data (dict): Feature importance data
            model_name (str): Name of the model
            output_file (str, optional): Path to save the report
            
        Returns:
            str: Report text
        """
        self.logger.info(f"Generating feature importance report for {model_name}")
        
        try:
            # Start with report header
            report = f"# Feature Importance Report: {model_name}\n\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add method information
            if 'method' in importance_data:
                method = importance_data['method']
                report += f"## Method: {method.replace('_', ' ').title()}\n\n"
            else:
                report += f"## Feature Importance Analysis\n\n"
            
            # Add timestamp if available
            if 'timestamp' in importance_data:
                report += f"Analysis performed: {importance_data['timestamp']}\n\n"
            
            # Get feature ranking
            feature_ranking = self.get_feature_ranking(importance_data)
            
            # Top features
            report += "## Top Features\n\n"
            report += "| Rank | Feature | Importance |\n"
            report += "| ---- | ------- | ---------- |\n"
            
            for i, (feature, importance) in enumerate(feature_ranking[:20]):  # Show top 20
                report += f"| {i+1} | {feature} | {importance:.6f} |\n"
            
            # Bottom features
            if len(feature_ranking) > 30:  # Only show if we have many features
                report += "\n## Bottom Features\n\n"
                report += "| Rank | Feature | Importance |\n"
                report += "| ---- | ------- | ---------- |\n"
                
                for i, (feature, importance) in enumerate(reversed(feature_ranking[-10:])):  # Show bottom 10
                    rank = len(feature_ranking) - i
                    report += f"| {rank} | {feature} | {importance:.6f} |\n"
            
            # Additional information based on method
            if 'method' in importance_data:
                method = importance_data['method']
                
                if method == 'permutation_importance':
                    report += "\n## Permutation Importance Details\n\n"
                    report += f"* Number of permutation repeats: {importance_data.get('n_repeats', 'N/A')}\n"
                    
                    # Add standard deviation information
                    if 'std_importance' in importance_data:
                        report += "\n### Feature Stability (Standard Deviation)\n\n"
                        report += "| Feature | Mean Importance | Std. Dev. | Coefficient of Variation |\n"
                        report += "| ------- | --------------- | --------- | ------------------------ |\n"
                        
                        for feature, importance in feature_ranking[:10]:  # Top 10 features
                            std = importance_data['std_importance'].get(feature, 0)
                            cv = std / importance if importance > 0 else 0
                            report += f"| {feature} | {importance:.6f} | {std:.6f} | {cv:.4f} |\n"
                
                elif method == 'shap_values':
                    report += "\n## SHAP Value Details\n\n"
                    report += f"* Sample size used: {importance_data.get('sample_size', 'N/A')}\n"
                    report += "* SHAP values represent the contribution of each feature to model predictions\n"
                
                elif method == 'partial_dependence':
                    report += "\n## Partial Dependence Details\n\n"
                    report += f"* Grid resolution: {importance_data.get('grid_resolution', 'N/A')}\n"
                    report += "* Partial dependence plots show how features affect predictions on average\n"
            
            # Save report
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
                
                self.logger.info(f"Saved feature importance report to {output_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating feature importance report: {str(e)}")
            return f"Error generating report: {str(e)}"
