"""
Feature Selector Module

This module provides tools for automatic feature selection based on importance metrics,
correlation analysis, and model performance, helping to identify the most relevant features
and reduce dimensionality.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_selection import SelectFromModel, RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

class FeatureSelector:
    """
    Provides methods for automatic feature selection to identify the most relevant features.
    
    Features:
    - Importance-based feature selection
    - Correlation-based feature removal
    - Recursive feature elimination
    - Sequential feature selection
    - Cross-validation based selection
    """
    
    def __init__(self, config=None):
        """
        Initialize the FeatureSelector.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.results_dir = os.path.join("data", "feature_importance", "selection")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default parameters
        self.n_jobs = self.config.get("n_jobs", -1)
        self.random_state = self.config.get("random_state", 42)
        self.cv_folds = self.config.get("cv_folds", 5)
        
        self.logger.info("FeatureSelector initialized")
    
    def select_by_importance(self, X, feature_names, importance_values, threshold=None, k=None):
        """
        Select features based on importance values.
        
        Args:
            X (numpy.ndarray): Feature matrix
            feature_names (list): Names of features
            importance_values (dict or array-like): Feature importance values
            threshold (float, optional): Minimum importance threshold
            k (int, optional): Number of top features to select
            
        Returns:
            tuple: Selected feature indices, selected feature names, and mask
        """
        self.logger.info("Selecting features by importance")
        
        try:
            # Convert importance_values to array if it's a dictionary
            if isinstance(importance_values, dict):
                # Ensure all features have importance values
                importance_array = np.zeros(len(feature_names))
                for i, feature in enumerate(feature_names):
                    importance_array[i] = importance_values.get(feature, 0)
            else:
                importance_array = np.array(importance_values)
            
            # Ensure importance array has correct shape
            if len(importance_array) != len(feature_names):
                self.logger.error(f"Importance array length ({len(importance_array)}) does not match feature count ({len(feature_names)})")
                return [], [], []
            
            # Determine selection method
            if k is not None:
                # Select top k features
                k = min(k, len(feature_names))
                selected_indices = np.argsort(importance_array)[-k:]
            elif threshold is not None:
                # Select features above threshold
                selected_indices = np.where(importance_array >= threshold)[0]
            else:
                # Default: select features with non-zero importance
                selected_indices = np.where(importance_array > 0)[0]
            
            # Create selection mask
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[selected_indices] = True
            
            # Get selected feature names
            selected_features = [feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Selected {len(selected_features)} features by importance")
            
            return selected_indices, selected_features, mask
            
        except Exception as e:
            self.logger.error(f"Error selecting features by importance: {str(e)}")
            return [], [], []
    
    def remove_correlated_features(self, X, feature_names, threshold=0.9, keep_higher_importance=True, importance_values=None):
        """
        Remove highly correlated features.
        
        Args:
            X (numpy.ndarray): Feature matrix
            feature_names (list): Names of features
            threshold (float): Correlation threshold
            keep_higher_importance (bool): Whether to keep the more important feature in correlated pairs
            importance_values (dict or array-like, optional): Feature importance values
            
        Returns:
            tuple: Selected feature indices, selected feature names, and mask
        """
        self.logger.info(f"Removing correlated features with threshold {threshold}")
        
        try:
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(X.T)
            
            # Convert importance_values to array if provided as dictionary
            if isinstance(importance_values, dict):
                importance_array = np.zeros(len(feature_names))
                for i, feature in enumerate(feature_names):
                    importance_array[i] = importance_values.get(feature, 0)
            elif importance_values is not None:
                importance_array = np.array(importance_values)
            else:
                # Use feature variances as proxy for importance if not provided
                importance_array = np.var(X, axis=0)
            
            # Initialize mask (1 = keep, 0 = remove)
            mask = np.ones(len(feature_names), dtype=bool)
            
            # Find features to remove
            for i in range(len(feature_names)):
                # Skip if already marked for removal
                if not mask[i]:
                    continue
                
                for j in range(i+1, len(feature_names)):
                    # Skip if already marked for removal
                    if not mask[j]:
                        continue
                    
                    # Check if correlation is above threshold
                    if abs(correlation_matrix[i, j]) >= threshold:
                        # Determine which feature to keep
                        if keep_higher_importance:
                            if importance_array[i] >= importance_array[j]:
                                # Keep i, remove j
                                mask[j] = False
                            else:
                                # Keep j, remove i
                                mask[i] = False
                                break  # Break inner loop as i is now removed
                        else:
                            # Always remove the second feature
                            mask[j] = False
            
            # Get selected feature indices and names
            selected_indices = np.where(mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            removed_count = len(feature_names) - len(selected_features)
            self.logger.info(f"Removed {removed_count} correlated features, kept {len(selected_features)} features")
            
            return selected_indices, selected_features, mask
            
        except Exception as e:
            self.logger.error(f"Error removing correlated features: {str(e)}")
            return [], [], []
    
    def select_with_model_based(self, X, y, feature_names, model=None, is_regression=False, threshold=None, k=None):
        """
        Select features using a model's built-in feature importance.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list): Names of features
            model (object, optional): Trained model with feature_importances_
            is_regression (bool): Whether this is a regression task
            threshold (float, optional): Importance threshold for selection
            k (int, optional): Number of top features to select
            
        Returns:
            tuple: Selected feature indices, selected feature names, and mask
        """
        self.logger.info("Selecting features using model-based selection")
        
        try:
            # Create model if not provided
            if model is None:
                if is_regression:
                    model = RandomForestRegressor(
                        n_estimators=100, 
                        random_state=self.random_state,
                        n_jobs=self.n_jobs
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=self.random_state,
                        n_jobs=self.n_jobs
                    )
                
                # Fit model
                model.fit(X, y)
            
            # Check if model has feature_importances_ attribute
            if not hasattr(model, 'feature_importances_'):
                self.logger.error("Model does not have feature_importances_ attribute")
                return [], [], []
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create SelectFromModel transformer
            if threshold is not None:
                selector = SelectFromModel(model, threshold=threshold, prefit=True)
            elif k is not None:
                # Use threshold that selects top k features
                sorted_importances = np.sort(importances)
                threshold = sorted_importances[-k] if k <= len(sorted_importances) else 0
                selector = SelectFromModel(model, threshold=threshold, prefit=True)
            else:
                # Use mean importance as threshold (default behavior)
                selector = SelectFromModel(model, prefit=True)
            
            # Get selection mask
            mask = selector.get_support()
            
            # Get selected feature indices and names
            selected_indices = np.where(mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Selected {len(selected_features)} features with model-based selection")
            
            return selected_indices, selected_features, mask
            
        except Exception as e:
            self.logger.error(f"Error in model-based feature selection: {str(e)}")
            return [], [], []
    
    def select_with_rfe(self, X, y, feature_names, n_features_to_select, is_regression=False, step=1):
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list): Names of features
            n_features_to_select (int): Target number of features to select
            is_regression (bool): Whether this is a regression task
            step (int): Number of features to remove at each iteration
            
        Returns:
            tuple: Selected feature indices, selected feature names, and mask
        """
        self.logger.info(f"Selecting {n_features_to_select} features using RFE")
        
        try:
            # Create estimator
            if is_regression:
                estimator = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            
            # Create RFE selector
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=step,
                verbose=0
            )
            
            # Fit RFE
            rfe.fit(X, y)
            
            # Get selection mask
            mask = rfe.support_
            
            # Get selected feature indices and names
            selected_indices = np.where(mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Get ranking
            ranking = rfe.ranking_
            
            self.logger.info(f"Selected {len(selected_features)} features with RFE")
            
            # Return with additional ranking information
            return selected_indices, selected_features, mask, ranking
            
        except Exception as e:
            self.logger.error(f"Error in RFE feature selection: {str(e)}")
            return [], [], [], []
    
    def select_with_sequential(self, X, y, feature_names, n_features_to_select, is_regression=False, direction='forward'):
        """
        Select features using Sequential Feature Selection.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list): Names of features
            n_features_to_select (int): Target number of features to select
            is_regression (bool): Whether this is a regression task
            direction (str): Direction of selection ('forward' or 'backward')
            
        Returns:
            tuple: Selected feature indices, selected feature names, and mask
        """
        self.logger.info(f"Selecting {n_features_to_select} features using {direction} sequential selection")
        
        try:
            # Create estimator (using lighter models for SFS as it's computationally intensive)
            if is_regression:
                estimator = Lasso(alpha=0.01, random_state=self.random_state)
            else:
                estimator = LogisticRegression(
                    C=1.0, 
                    penalty='l2',
                    solver='saga',
                    max_iter=1000,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            
            # Create Sequential Feature Selector
            sfs = SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                direction=direction,
                scoring='r2' if is_regression else 'accuracy',
                cv=min(self.cv_folds, len(y) // 2),
                n_jobs=self.n_jobs
            )
            
            # Fit SFS
            sfs.fit(X, y)
            
            # Get selection mask
            mask = sfs.get_support()
            
            # Get selected feature indices and names
            selected_indices = np.where(mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Selected {len(selected_features)} features with sequential selection")
            
            return selected_indices, selected_features, mask
            
        except Exception as e:
            self.logger.error(f"Error in sequential feature selection: {str(e)}")
            return [], [], []
    
    def evaluate_feature_subsets(self, X, y, feature_names, is_regression=False, min_features=1, max_features=None, step=1):
        """
        Evaluate model performance with different feature subset sizes.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list): Names of features
            is_regression (bool): Whether this is a regression task
            min_features (int): Minimum number of features to consider
            max_features (int, optional): Maximum number of features to consider
            step (int): Step size for feature count
            
        Returns:
            dict: Performance results for different feature subset sizes
        """
        self.logger.info("Evaluating feature subsets")
        
        try:
            # Set default max_features if not provided
            if max_features is None:
                max_features = len(feature_names)
            
            # Ensure valid feature ranges
            min_features = max(1, min(min_features, len(feature_names)))
            max_features = max(min_features, min(max_features, len(feature_names)))
            
            # Create estimator with built-in feature importance
            if is_regression:
                estimator = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                scoring = 'r2'
            else:
                estimator = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                scoring = 'accuracy'
            
            # Train initial model to get feature importances
            estimator.fit(X, y)
            importances = estimator.feature_importances_
            
            # Sort features by importance (descending)
            sorted_indices = np.argsort(importances)[::-1]
            
            # Evaluate performance for different feature subset sizes
            results = {}
            
            for n_features in range(min_features, max_features + 1, step):
                # Get top n_features
                features_to_use = sorted_indices[:n_features]
                X_subset = X[:, features_to_use]
                
                # Evaluate with cross-validation
                cv_scores = cross_val_score(
                    estimator, X_subset, y, 
                    cv=min(self.cv_folds, len(y) // 2),
                    scoring=scoring, 
                    n_jobs=self.n_jobs
                )
                
                # Store results
                results[n_features] = {
                    'mean_score': np.mean(cv_scores),
                    'std_score': np.std(cv_scores),
                    'feature_indices': features_to_use.tolist(),
                    'feature_names': [feature_names[i] for i in features_to_use]
                }
                
            self.logger.info(f"Evaluated {len(results)} feature subsets")
            
            # Find optimal feature count
            optimal_n = max(results.keys(), key=lambda k: results[k]['mean_score'])
            results['optimal_n_features'] = optimal_n
            results['optimal_features'] = results[optimal_n]['feature_names']
            results['optimal_score'] = results[optimal_n]['mean_score']
            
            self.logger.info(f"Optimal feature count: {optimal_n} with score: {results[optimal_n]['mean_score']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating feature subsets: {str(e)}")
            return {}
    
    def find_redundant_features(self, X, y, feature_names, is_regression=False):
        """
        Find redundant features by evaluating the impact of their removal.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list): Names of features
            is_regression (bool): Whether this is a regression task
            
        Returns:
            dict: Results of redundancy analysis
        """
        self.logger.info("Finding redundant features")
        
        try:
            # Create base estimator
            if is_regression:
                estimator = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            
            # Get baseline performance (using all features)
            estimator.fit(X, y)
            y_pred = estimator.predict(X)
            
            if is_regression:
                baseline_score = r2_score(y, y_pred)
                baseline_error = mean_squared_error(y, y_pred, squared=False)  # RMSE
            else:
                baseline_score = accuracy_score(y, y_pred)
                baseline_error = 1 - baseline_score
            
            # Evaluate the impact of removing each feature
            results = []
            
            for i in range(len(feature_names)):
                # Create dataset without this feature
                X_without_feature = np.delete(X, i, axis=1)
                feature_names_without = [f for j, f in enumerate(feature_names) if j != i]
                
                # Train and evaluate model
                estimator.fit(X_without_feature, y)
                y_pred = estimator.predict(X_without_feature)
                
                if is_regression:
                    score = r2_score(y, y_pred)
                    error = mean_squared_error(y, y_pred, squared=False)  # RMSE
                else:
                    score = accuracy_score(y, y_pred)
                    error = 1 - score
                
                # Calculate performance impact
                score_change = score - baseline_score
                error_change = error - baseline_error
                
                # Store results
                results.append({
                    'feature': feature_names[i],
                    'score_without': score,
                    'score_change': score_change,
                    'error_without': error,
                    'error_change': error_change,
                    'is_redundant': score_change >= 0  # Feature is redundant if removing it doesn't hurt (or improves) performance
                })
            
            # Sort by score change (ascending, so most negative change = most important feature first)
            results = sorted(results, key=lambda x: x['score_change'])
            
            # Identify redundant features
            redundant_features = [r['feature'] for r in results if r['is_redundant']]
            
            # Compile analysis
            analysis = {
                'baseline_score': baseline_score,
                'baseline_error': baseline_error,
                'feature_analysis': results,
                'redundant_features': redundant_features,
                'redundant_count': len(redundant_features),
                'essential_count': len(feature_names) - len(redundant_features)
            }
            
            self.logger.info(f"Identified {len(redundant_features)} potentially redundant features")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error finding redundant features: {str(e)}")
            return {}
    
    def optimize_feature_selection(self, X, y, feature_names, is_regression=False, optimization_metric='auto', max_iterations=10):
        """
        Optimize feature selection using a hybrid approach.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            feature_names (list): Names of features
            is_regression (bool): Whether this is a regression task
            optimization_metric (str): Metric to optimize ('accuracy', 'r2', 'rmse', 'auto')
            max_iterations (int): Maximum number of selection iterations
            
        Returns:
            dict: Optimized feature selection results
        """
        self.logger.info("Optimizing feature selection")
        
        try:
            # Set default optimization metric if 'auto'
            if optimization_metric == 'auto':
                optimization_metric = 'r2' if is_regression else 'accuracy'
            
            # Initialize with all features
            current_mask = np.ones(len(feature_names), dtype=bool)
            best_mask = current_mask.copy()
            
            # Create estimator
            if is_regression:
                estimator = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            
            # Evaluate baseline performance
            estimator.fit(X, y)
            y_pred = estimator.predict(X)
            
            if optimization_metric == 'accuracy':
                best_score = accuracy_score(y, y_pred)
            elif optimization_metric == 'r2':
                best_score = r2_score(y, y_pred)
            elif optimization_metric == 'rmse':
                best_score = -mean_squared_error(y, y_pred, squared=False)  # Negative RMSE for maximization
            else:
                self.logger.warning(f"Unknown optimization metric: {optimization_metric}, using accuracy")
                best_score = accuracy_score(y, y_pred)
            
            # Get feature importances
            importances = estimator.feature_importances_
            
            # Iterative feature selection process
            iteration_results = []
            
            for iteration in range(max_iterations):
                # Step 1: Remove highly correlated features
                X_current = X[:, current_mask]
                current_feature_names = [feature_names[i] for i, use in enumerate(current_mask) if use]
                
                # Calculate correlation matrix
                if X_current.shape[1] > 1:  # Need at least 2 features for correlation
                    correlation_matrix = np.corrcoef(X_current.T)
                    
                    # Find the most correlated pair
                    corr_threshold = 0.9 - (iteration * 0.1)  # Gradually reduce threshold
                    corr_threshold = max(0.5, corr_threshold)  # Don't go below 0.5
                    
                    highest_corr = 0
                    corr_pair = None
                    
                    for i in range(correlation_matrix.shape[0]):
                        for j in range(i+1, correlation_matrix.shape[1]):
                            if abs(correlation_matrix[i, j]) > highest_corr:
                                highest_corr = abs(correlation_matrix[i, j])
                                corr_pair = (i, j)
                    
                    # Remove one of the correlated features if above threshold
                    if highest_corr >= corr_threshold and corr_pair is not None:
                        i, j = corr_pair
                        
                        # Get global indices
                        global_indices = np.where(current_mask)[0]
                        feature_i = global_indices[i]
                        feature_j = global_indices[j]
                        
                        # Remove feature with lower importance
                        if importances[feature_i] <= importances[feature_j]:
                            current_mask[feature_i] = False
                        else:
                            current_mask[feature_j] = False
                
                # Step 2: Evaluate current feature set
                X_current = X[:, current_mask]
                
                # Cross-validate to get more reliable score
                cv_scores = cross_val_score(
                    estimator, X_current, y, 
                    cv=min(self.cv_folds, len(y) // 2),
                    scoring=optimization_metric, 
                    n_jobs=self.n_jobs
                )
                
                current_score = np.mean(cv_scores)
                
                # Update best if improved
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask.copy()
                
                # Store iteration results
                current_feature_names = [feature_names[i] for i, use in enumerate(current_mask) if use]
                iteration_results.append({
                    'iteration': iteration + 1,
                    'score': current_score,
                    'n_features': sum(current_mask),
                    'feature_names': current_feature_names
                })
                
                self.logger.info(f"Iteration {iteration+1}: {sum(current_mask)} features, score: {current_score:.4f}")
                
                # Early stopping if we've removed too many features
                if sum(current_mask) < 2:
                    break
            
            # Get final feature set
            selected_indices = np.where(best_mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Compile results
            results = {
                'selected_indices': selected_indices.tolist(),
                'selected_features': selected_features,
                'best_score': best_score,
                'n_features': len(selected_features),
                'original_n_features': len(feature_names),
                'feature_reduction': 1 - (len(selected_features) / len(feature_names)),
                'optimization_metric': optimization_metric,
                'iterations': iteration_results
            }
            
            self.logger.info(f"Optimized selection: {len(selected_features)} features with score: {best_score:.4f}")
            
            return results, best_mask
            
        except Exception as e:
            self.logger.error(f"Error optimizing feature selection: {str(e)}")
            return {}, np.ones(len(feature_names), dtype=bool)
    
    def save_selection_results(self, results, model_name, selection_method, output_file=None):
        """
        Save feature selection results to a file.
        
        Args:
            results (dict): Feature selection results
            model_name (str): Name of the model
            selection_method (str): Method used for selection
            output_file (str, optional): Path to save the results
            
        Returns:
            str: Path to the saved file
        """
        # Create a results dictionary with metadata
        save_data = {
            'model_name': model_name,
            'selection_method': selection_method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }
        
        # Create filename if not provided
        if output_file is None:
            safe_name = model_name.replace(' ', '_').lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{safe_name}_{selection_method}_selection_{timestamp}.json"
            output_file = os.path.join(self.results_dir, filename)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Saved selection results to {output_file}")
        
        return output_file
    
    def load_selection_results(self, filepath):
        """
        Load feature selection results from a file.
        
        Args:
            filepath (str): Path to the results file
            
        Returns:
            dict: Feature selection results
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded selection results from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading selection results: {str(e)}")
            return None
    
    def generate_selection_report(self, results, model_name, output_file=None):
        """
        Generate a human-readable report of feature selection findings.
        
        Args:
            results (dict): Feature selection results
            model_name (str): Name of the model
            output_file (str, optional): Path to save the report
            
        Returns:
            str: Report text
        """
        self.logger.info(f"Generating feature selection report for {model_name}")
        
        try:
            # Start with report header
            report = f"# Feature Selection Report: {model_name}\n\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add method information
            selection_method = results.get('selection_method', 'Unknown Method')
            report += f"## Method: {selection_method.replace('_', ' ').title()}\n\n"
            
            # Add summary statistics
            if 'n_features' in results and 'original_n_features' in results:
                n_features = results['n_features']
                original_n_features = results['original_n_features']
                reduction = results.get('feature_reduction', 1 - (n_features / original_n_features))
                
                report += f"* Selected {n_features} features from {original_n_features} original features\n"
                report += f"* Reduced feature dimensions by {reduction:.1%}\n"
                
                if 'best_score' in results:
                    metric = results.get('optimization_metric', 'score')
                    report += f"* Best {metric} score: {results['best_score']:.4f}\n"
                
                report += "\n"
            
            # List selected features
            if 'selected_features' in results:
                report += "## Selected Features\n\n"
                
                for i, feature in enumerate(results['selected_features']):
                    report += f"{i+1}. {feature}\n"
                
                report += "\n"
            
            # Add iteration results if available
            if 'iterations' in results:
                report += "## Iteration Results\n\n"
                report += "| Iteration | Features | Score |\n"
                report += "| --------- | -------- | ----- |\n"
                
                for iteration in results['iterations']:
                    report += f"| {iteration['iteration']} | {iteration['n_features']} | {iteration['score']:.4f} |\n"
                
                report += "\n"
            
            # Add redundant features if available
            if 'redundant_features' in results:
                report += "## Redundant Features\n\n"
                
                for feature in results['redundant_features']:
                    report += f"* {feature}\n"
                
                report += "\n"
            
            # Add feature analysis if available
            if 'feature_analysis' in results:
                report += "## Feature Importance Analysis\n\n"
                report += "| Feature | Impact on Score | Redundant? |\n"
                report += "| ------- | --------------- | ---------- |\n"
                
                for analysis in results['feature_analysis']:
                    redundant = "Yes" if analysis.get('is_redundant', False) else "No"
                    report += f"| {analysis['feature']} | {analysis['score_change']:.4f} | {redundant} |\n"
                
                report += "\n"
            
            # Save report
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
                
                self.logger.info(f"Saved feature selection report to {output_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating feature selection report: {str(e)}")
            return f"Error generating report: {str(e)}"
