"""
Adaptive Weighting Module

This module implements a dynamic weighting system for the meta-learning component,
adjusting the weight given to different models based on their historical performance
in similar market conditions.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class AdaptiveWeighting:
    """
    Dynamically adjusts model weights based on historical performance in
    similar market conditions. Uses a combination of performance tracking,
    market regime detection, and environmental similarity to optimize weights.
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the AdaptiveWeighting system.
        
        Args:
            config (dict): Configuration dictionary
            drive_manager: Google Drive manager for data storage/retrieval
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("meta_learning", {}).get("adaptive_weighting", {})
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.momentum = self.config.get("momentum", 0.8)
        self.recency_weight = self.config.get("recency_weight", 0.7)
        self.history_window = self.config.get("history_window", 60)  # days
        self.min_observations = self.config.get("min_observations", 5)
        self.max_weight = self.config.get("max_weight", 0.8)
        self.min_weight = self.config.get("min_weight", 0.05)
        self.num_clusters = self.config.get("num_clusters", 5)
        
        # Internal state
        self.model_names = []
        self.model_weights = {}
        self.performance_history = defaultdict(list)
        self.weight_updates = defaultdict(float)  # For momentum
        self.market_clusters = None
        self.market_scaler = StandardScaler()
        
        # Ensure directory exists
        os.makedirs("data/meta_learning", exist_ok=True)
        
        # Load saved state if available
        self._load_state()
        
        self.logger.info(f"AdaptiveWeighting initialized with {len(self.model_names)} models")
    
    def _load_state(self):
        """Load model weights and history from disk or Google Drive."""
        try:
            # Try local file first
            state_file = "data/meta_learning/adaptive_weighting_state.json"
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                
                self.model_names = state.get("model_names", [])
                self.model_weights = state.get("model_weights", {})
                self.performance_history = defaultdict(list)
                
                # Convert performance history from JSON
                for model, history in state.get("performance_history", {}).items():
                    self.performance_history[model] = history
                
                self.logger.info(f"Loaded adaptive weighting state with {len(self.model_names)} models")
                
                # Load clustering model if available
                if os.path.exists("data/meta_learning/market_clusters.npz"):
                    data = np.load("data/meta_learning/market_clusters.npz", allow_pickle=True)
                    self.market_clusters = KMeans(n_clusters=self.num_clusters)
                    self.market_clusters.cluster_centers_ = data["centers"]
                    self.market_clusters.labels_ = data["labels"]
                    
                    # Load scaler params
                    if os.path.exists("data/meta_learning/market_scaler.json"):
                        with open("data/meta_learning/market_scaler.json", "r") as f:
                            scaler_params = json.load(f)
                        self.market_scaler.mean_ = np.array(scaler_params["mean"])
                        self.market_scaler.scale_ = np.array(scaler_params["scale"])
                        self.market_scaler.var_ = np.array(scaler_params["var"])
                        self.market_scaler.n_features_in_ = len(scaler_params["mean"])
                    
            elif self.drive_manager and self.drive_manager.file_exists("meta_learning/adaptive_weighting_state.json"):
                # Try Google Drive if local file not found
                content = self.drive_manager.download_file("meta_learning/adaptive_weighting_state.json")
                state = json.loads(content)
                
                self.model_names = state.get("model_names", [])
                self.model_weights = state.get("model_weights", {})
                self.performance_history = defaultdict(list)
                
                # Convert performance history from JSON
                for model, history in state.get("performance_history", {}).items():
                    self.performance_history[model] = history
                
                # Save locally for future use
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)
                
                self.logger.info(f"Loaded adaptive weighting state from Google Drive with {len(self.model_names)} models")
        
        except Exception as e:
            self.logger.error(f"Error loading adaptive weighting state: {str(e)}")
            self.logger.info("Starting with default weights")
    
    def _save_state(self):
        """Save model weights and history to disk and Google Drive."""
        try:
            state = {
                "model_names": self.model_names,
                "model_weights": self.model_weights,
                "performance_history": dict(self.performance_history),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to local file
            state_file = "data/meta_learning/adaptive_weighting_state.json"
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            
            # Save to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "meta_learning/adaptive_weighting_state.json",
                    json.dumps(state, indent=2),
                    mime_type="application/json"
                )
            
            # Save clustering model if available
            if self.market_clusters is not None:
                np.savez(
                    "data/meta_learning/market_clusters.npz",
                    centers=self.market_clusters.cluster_centers_,
                    labels=self.market_clusters.labels_
                )
                
                # Save scaler params
                scaler_params = {
                    "mean": self.market_scaler.mean_.tolist(),
                    "scale": self.market_scaler.scale_.tolist(),
                    "var": self.market_scaler.var_.tolist()
                }
                with open("data/meta_learning/market_scaler.json", "w") as f:
                    json.dump(scaler_params, f, indent=2)
            
            self.logger.info("Saved adaptive weighting state")
            
        except Exception as e:
            self.logger.error(f"Error saving adaptive weighting state: {str(e)}")
    
    def register_models(self, model_names):
        """
        Register models to be weighted.
        
        Args:
            model_names (list): List of model names
        """
        # Add any new models
        for model in model_names:
            if model not in self.model_names:
                self.model_names.append(model)
                self.model_weights[model] = 1.0 / len(model_names)  # Equal weights initially
                self.performance_history[model] = []
        
        # Normalize weights
        self._normalize_weights()
        
        self.logger.info(f"Registered models: {', '.join(self.model_names)}")
    
    def _normalize_weights(self):
        """Normalize weights to ensure they sum to 1."""
        if not self.model_weights:
            return
            
        # Apply min/max constraints
        for model in self.model_weights:
            self.model_weights[model] = max(self.min_weight, min(self.max_weight, self.model_weights[model]))
        
        # Normalize
        weight_sum = sum(self.model_weights.values())
        if weight_sum > 0:
            for model in self.model_weights:
                self.model_weights[model] /= weight_sum
    
    def update_weights(self, performance_metrics, market_features):
        """
        Update model weights based on recent performance.
        
        Args:
            performance_metrics (dict): Dict mapping model names to performance metrics
            market_features (dict): Current market features for regime identification
            
        Returns:
            dict: Updated model weights
        """
        if not self.model_names:
            self.logger.warning("No models registered for weighting")
            return {}
        
        # Add performance to history with timestamp
        timestamp = datetime.now()
        for model, metrics in performance_metrics.items():
            if model in self.model_names:
                entry = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": metrics,
                    "market_features": market_features
                }
                self.performance_history[model].append(entry)
        
        # Find similar historical market conditions
        similar_periods = self._find_similar_periods(market_features)
        
        # Calculate performance in similar conditions
        performance_in_similar = {}
        for model in self.model_names:
            performance_in_similar[model] = self._calculate_performance_in_periods(model, similar_periods)
        
        # Update weights based on relative performance
        self._update_weights_based_on_performance(performance_in_similar)
        
        # Save updated state
        self._save_state()
        
        return self.get_weights()
    
    def _find_similar_periods(self, current_features):
        """
        Find historical periods with similar market conditions.
        
        Args:
            current_features (dict): Current market feature values
            
        Returns:
            list: List of timestamps for similar historical periods
        """
        # Extract features into a format suitable for clustering
        feature_keys = sorted(current_features.keys())
        current_vector = np.array([current_features[k] for k in feature_keys]).reshape(1, -1)
        
        # Prepare historical feature vectors
        historical_vectors = []
        timestamps = []
        
        # Collect historical data points from all models
        for model in self.model_names:
            for entry in self.performance_history[model]:
                if "market_features" in entry:
                    hist_features = entry["market_features"]
                    if all(k in hist_features for k in feature_keys):
                        vector = [hist_features[k] for k in feature_keys]
                        historical_vectors.append(vector)
                        timestamps.append(entry["timestamp"])
        
        # If not enough historical data, return all periods
        if len(historical_vectors) < self.min_observations:
            all_periods = []
            for model in self.model_names:
                all_periods.extend([entry["timestamp"] for entry in self.performance_history[model]])
            return all_periods
        
        # Prepare data for clustering
        historical_matrix = np.array(historical_vectors)
        
        # Fit or update the clustering model
        all_data = np.vstack([historical_matrix, current_vector])
        
        # Scale the data
        if not hasattr(self.market_scaler, 'mean_'):
            scaled_data = self.market_scaler.fit_transform(all_data)
        else:
            scaled_data = self.market_scaler.transform(all_data)
        
        if self.market_clusters is None or len(historical_vectors) % 10 == 0:  # Periodically retrain
            self.market_clusters = KMeans(n_clusters=min(self.num_clusters, len(historical_vectors)))
            self.market_clusters.fit(scaled_data[:-1])  # Fit on historical data only
        
        # Get cluster for current market state
        current_cluster = self.market_clusters.predict(scaled_data[-1].reshape(1, -1))[0]
        
        # Find all historical periods in same cluster
        cluster_assignments = self.market_clusters.predict(scaled_data[:-1])
        similar_indices = [i for i, c in enumerate(cluster_assignments) if c == current_cluster]
        
        # Return timestamps for similar periods
        return [timestamps[i] for i in similar_indices]
    
    def _calculate_performance_in_periods(self, model, periods):
        """
        Calculate model performance during similar historical periods.
        
        Args:
            model (str): Model name
            periods (list): List of timestamps for relevant periods
            
        Returns:
            float: Weighted average performance
        """
        if not periods:
            return 0.0
            
        # Extract performance for the given periods
        performances = []
        timestamps = []
        
        for entry in self.performance_history[model]:
            if entry["timestamp"] in periods:
                # Get primary performance metric (e.g., profit, accuracy)
                metric = entry["metrics"].get("profit", 
                        entry["metrics"].get("pnl",
                        entry["metrics"].get("accuracy", 0)))
                
                performances.append(metric)
                timestamps.append(datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S"))
        
        if not performances:
            return 0.0
        
        # Apply recency weighting if timestamps exist
        if timestamps:
            # Calculate recency weights (more recent = higher weight)
            now = datetime.now()
            max_age = max((now - min(timestamps)).total_seconds(), 1)
            
            recency_weights = [
                1.0 - self.recency_weight * ((now - ts).total_seconds() / max_age)
                for ts in timestamps
            ]
            
            # Calculate weighted average
            return np.average(performances, weights=recency_weights)
        else:
            return np.mean(performances)
    
    def _update_weights_based_on_performance(self, performances):
        """
        Update model weights based on relative performance.
        
        Args:
            performances (dict): Dict mapping model names to performance metrics
        """
        # Skip if no performances available
        if not performances or all(p == 0 for p in performances.values()):
            return
        
        # Calculate relative performance (min-max normalization)
        min_perf = min(performances.values())
        max_perf = max(performances.values())
        
        if max_perf == min_perf:
            # All equal performance, keep weights as is
            return
            
        relative_performances = {}
        for model, perf in performances.items():
            relative_performances[model] = (perf - min_perf) / (max_perf - min_perf)
        
        # Update weights with momentum (to avoid rapid changes)
        for model in self.model_names:
            rel_perf = relative_performances.get(model, 0)
            
            # Calculate update with momentum
            update = self.learning_rate * (rel_perf - self.model_weights[model])
            self.weight_updates[model] = self.momentum * self.weight_updates[model] + (1 - self.momentum) * update
            
            # Apply update
            self.model_weights[model] += self.weight_updates[model]
        
        # Normalize the weights
        self._normalize_weights()
        
        self.logger.info(f"Updated model weights: {', '.join([f'{m}:{w:.3f}' for m, w in self.model_weights.items()])}")
    
    def get_weights(self):
        """
        Get current model weights.
        
        Returns:
            dict: Current model weights
        """
        return self.model_weights.copy()
    
    def get_model_performance_history(self, model_name):
        """
        Get performance history for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            list: List of historical performance entries
        """
        if model_name in self.performance_history:
            return self.performance_history[model_name].copy()
        return []
    
    def clear_old_history(self, days_to_keep=None):
        """
        Clear old history beyond specified window.
        
        Args:
            days_to_keep (int, optional): Number of days of history to retain
        """
        if days_to_keep is None:
            days_to_keep = self.history_window
            
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
        
        for model in self.model_names:
            self.performance_history[model] = [
                entry for entry in self.performance_history[model]
                if entry["timestamp"] > cutoff_str
            ]
        
        self.logger.info(f"Cleared performance history older than {days_to_keep} days")
