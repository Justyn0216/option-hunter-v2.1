"""
State Abstractor Module

This module creates abstract state representations for higher-level
decision making in the hierarchical reinforcement learning system.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import json

class StateAbstractor:
    """
    Creates abstract state representations for higher-level decision making.
    Reduces the dimensionality of complex market state representations
    to enable more efficient learning at the meta-controller level.
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the StateAbstractor.
        
        Args:
            config (dict): Configuration dictionary
            drive_manager: GoogleDriveManager for saving/loading models
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("hierarchical_rl", {}).get("state_abstractor", {})
        self.drive_manager = drive_manager
        
        # Extract configuration parameters
        self.abstract_state_dim = self.config.get("abstract_state_dim", 20)
        self.scaler_update_frequency = self.config.get("scaler_update_frequency", 100)
        self.pca_update_frequency = self.config.get("pca_update_frequency", 500)
        self.use_pca = self.config.get("use_pca", True)
        self.feature_selection = self.config.get("feature_selection", "auto")
        
        # Initialize state processing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.abstract_state_dim) if self.use_pca else None
        
        # Important feature indices
        self.important_features = self.config.get("important_features", [])
        
        # State tracking
        self.raw_state_buffer = []
        self.update_counter = 0
        
        # Directory for models
        self.model_dir = "data/hierarchical_rl"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
        
        self.logger.info(f"StateAbstractor initialized with {self.abstract_state_dim} dimensions")
    
    def _load_models(self):
        """Load saved scaler and PCA models if they exist."""
        try:
            scaler_path = f"{self.model_dir}/state_scaler.joblib"
            pca_path = f"{self.model_dir}/state_pca.joblib"
            features_path = f"{self.model_dir}/important_features.json"
            
            # Load scaler
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Loaded saved state scaler")
            elif self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/state_scaler.joblib"):
                # Download from Google Drive
                scaler_data = self.drive_manager.download_file_binary("hierarchical_rl/state_scaler.joblib")
                with open(scaler_path, 'wb') as f:
                    f.write(scaler_data)
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Loaded state scaler from Google Drive")
            
            # Load PCA if using
            if self.use_pca and os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
                self.logger.info("Loaded saved state PCA")
            elif self.use_pca and self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/state_pca.joblib"):
                # Download from Google Drive
                pca_data = self.drive_manager.download_file_binary("hierarchical_rl/state_pca.joblib")
                with open(pca_path, 'wb') as f:
                    f.write(pca_data)
                self.pca = joblib.load(pca_path)
                self.logger.info("Loaded state PCA from Google Drive")
            
            # Load important features
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.important_features = json.load(f)
                self.logger.info(f"Loaded {len(self.important_features)} important features")
            elif self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/important_features.json"):
                features_data = self.drive_manager.download_file("hierarchical_rl/important_features.json")
                self.important_features = json.loads(features_data)
                with open(features_path, 'w') as f:
                    f.write(features_data)
                self.logger.info(f"Loaded {len(self.important_features)} important features from Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error loading state abstractor models: {str(e)}")
            self.logger.info("Starting with new models")
    
    def _save_models(self):
        """Save scaler and PCA models."""
        try:
            scaler_path = f"{self.model_dir}/state_scaler.joblib"
            pca_path = f"{self.model_dir}/state_pca.joblib"
            features_path = f"{self.model_dir}/important_features.json"
            
            # Save scaler
            joblib.dump(self.scaler, scaler_path)
            
            # Save PCA if using
            if self.use_pca and self.pca is not None:
                joblib.dump(self.pca, pca_path)
            
            # Save important features
            with open(features_path, 'w') as f:
                json.dump(self.important_features, f)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open(scaler_path, 'rb') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/state_scaler.joblib",
                        f.read(),
                        mime_type="application/octet-stream"
                    )
                
                if self.use_pca and self.pca is not None:
                    with open(pca_path, 'rb') as f:
                        self.drive_manager.upload_file(
                            "hierarchical_rl/state_pca.joblib",
                            f.read(),
                            mime_type="application/octet-stream"
                        )
                
                with open(features_path, 'r') as f:
                    self.drive_manager.upload_file(
                        "hierarchical_rl/important_features.json",
                        f.read(),
                        mime_type="application/json"
                    )
            
            self.logger.info("Saved state abstractor models")
            
        except Exception as e:
            self.logger.error(f"Error saving state abstractor models: {str(e)}")
    
    def _convert_to_vector(self, state_features):
        """
        Convert state features dictionary to a flat vector.
        
        Args:
            state_features (dict): Dictionary of state features
            
        Returns:
            numpy.ndarray: Flat vector representation
        """
        # Check if we have important features defined
        if self.important_features and self.feature_selection != "auto":
            # Use only important features
            feature_values = []
            for feature in self.important_features:
                if feature in state_features:
                    value = state_features[feature]
                    # Handle different types of values
                    if isinstance(value, (int, float)):
                        feature_values.append(value)
                    elif isinstance(value, (list, np.ndarray)) and len(value) == 1:
                        feature_values.append(value[0])
                    elif isinstance(value, (list, np.ndarray)):
                        # For arrays, use the first few values
                        feature_values.extend(value[:min(5, len(value))])
                    else:
                        # Skip non-numeric features
                        feature_values.append(0.0)
                else:
                    # Feature not present, use default value
                    feature_values.append(0.0)
            
            return np.array(feature_values)
        else:
            # Use all available numeric features
            feature_values = []
            feature_names = []
            
            for key, value in state_features.items():
                # Skip non-numeric or complex nested features
                if isinstance(value, (int, float)):
                    feature_values.append(value)
                    feature_names.append(key)
                elif isinstance(value, (list, np.ndarray)) and len(value) == 1:
                    feature_values.append(value[0])
                    feature_names.append(key)
                elif isinstance(value, (list, np.ndarray)) and all(isinstance(x, (int, float)) for x in value):
                    # For arrays, use the first few values
                    for i, item in enumerate(value[:min(5, len(value))]):
                        feature_values.append(item)
                        feature_names.append(f"{key}_{i}")
            
            # If feature selection is "auto", update the important features periodically
            if self.feature_selection == "auto" and not self.important_features:
                self.important_features = feature_names
                self.logger.info(f"Auto-detected {len(self.important_features)} important features")
            
            return np.array(feature_values)
    
    def get_abstract_state(self, state_features):
        """
        Create an abstract state representation from raw state features.
        
        Args:
            state_features (dict): Dictionary of state features
            
        Returns:
            list: Abstract state representation
        """
        # Convert state to vector
        state_vector = self._convert_to_vector(state_features)
        
        # Add to buffer for model updates
        self.raw_state_buffer.append(state_vector)
        if len(self.raw_state_buffer) > 1000:
            self.raw_state_buffer = self.raw_state_buffer[-1000:]
        
        # Check if scaler has been fit
        if not hasattr(self.scaler, 'mean_'):
            # Initialize with identity transformation
            self.scaler.fit([state_vector])
        
        # Scale the vector
        scaled_vector = self.scaler.transform([state_vector])[0]
        
        # Update counter and check for model updates
        self.update_counter += 1
        self._check_model_updates()
        
        # Apply dimensionality reduction if using PCA
        if self.use_pca and self.pca is not None and hasattr(self.pca, 'components_'):
            # Transform to abstract state
            abstract_state = self.pca.transform([scaled_vector])[0]
        else:
            # If not using PCA or PCA not initialized yet, take first N dimensions
            abstract_state = scaled_vector[:self.abstract_state_dim]
            
            # Pad if necessary
            if len(abstract_state) < self.abstract_state_dim:
                padding = np.zeros(self.abstract_state_dim - len(abstract_state))
                abstract_state = np.concatenate([abstract_state, padding])
        
        return abstract_state.tolist()
    
    def _check_model_updates(self):
        """Check if models need to be updated based on new data."""
        # Update scaler periodically
        if self.update_counter % self.scaler_update_frequency == 0 and len(self.raw_state_buffer) > 10:
            try:
                self.logger.debug("Updating state scaler")
                self.scaler.fit(self.raw_state_buffer)
            except Exception as e:
                self.logger.error(f"Error updating scaler: {str(e)}")
        
        # Update PCA periodically
        if self.use_pca and self.update_counter % self.pca_update_frequency == 0 and len(self.raw_state_buffer) > self.abstract_state_dim:
            try:
                self.logger.debug("Updating state PCA")
                # Scale the buffer
                scaled_buffer = self.scaler.transform(self.raw_state_buffer)
                # Fit PCA
                self.pca.fit(scaled_buffer)
                # Save models after update
                self._save_models()
            except Exception as e:
                self.logger.error(f"Error updating PCA: {str(e)}")
    
    def update_important_features(self, new_features):
        """
        Update the list of important features.
        
        Args:
            new_features (list): List of important feature names
        """
        self.important_features = new_features
        self.logger.info(f"Updated important features list with {len(new_features)} features")
        self._save_models()
    
    def get_feature_importance(self):
        """
        Get feature importance information if available.
        
        Returns:
            dict: Feature importance information
        """
        if not self.use_pca or not hasattr(self.pca, 'components_'):
            return {"error": "PCA not initialized or not in use"}
        
        try:
            # Get feature importances from PCA components
            components = self.pca.components_
            explained_variance = self.pca.explained_variance_ratio_
            
            # Calculate absolute importance for each feature across components
            feature_importances = np.sum(np.abs(components.T * explained_variance), axis=1)
            
            # Normalize to sum to 1
            feature_importances = feature_importances / np.sum(feature_importances)
            
            # Create dictionary mapping feature names to importance
            importance_dict = {}
            
            # Map back to feature names if available
            if self.important_features and len(self.important_features) == len(feature_importances):
                for i, feature in enumerate(self.important_features):
                    importance_dict[feature] = float(feature_importances[i])
            else:
                # Use generic feature names
                for i, importance in enumerate(feature_importances):
                    importance_dict[f"feature_{i}"] = float(importance)
            
            # Sort by importance
            sorted_importances = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "feature_importances": sorted_importances,
                "explained_variance": self.pca.explained_variance_ratio_.tolist(),
                "total_explained_variance": float(sum(self.pca.explained_variance_ratio_))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {"error": str(e)}
    
    def is_initialized(self):
        """
        Check if the abstractor is properly initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        if not hasattr(self.scaler, 'mean_'):
            return False
            
        if self.use_pca and (self.pca is None or not hasattr(self.pca, 'components_')):
            return False
            
        return True
    
    def get_abstract_state_dim(self):
        """
        Get the dimension of the abstract state.
        
        Returns:
            int: Abstract state dimension
        """
        return self.abstract_state_dim
