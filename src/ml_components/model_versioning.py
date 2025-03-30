"""
Model Versioning Module

This module provides tools for tracking and managing ML model versions,
including version tracking, metadata storage, and model lifecycle management.
"""

import logging
import os
import json
import pickle
import shutil
import hashlib
import uuid
from datetime import datetime
import pandas as pd

class ModelVersioning:
    """
    Model versioning system for tracking and managing ML model versions.
    
    Features:
    - Model version tracking and history
    - Model metadata storage
    - Model deployment and rollback
    - Model comparison and selection
    """
    
    def __init__(self, config=None):
        """
        Initialize the ModelVersioning system.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.base_dir = "models"
        self.versions_dir = os.path.join(self.base_dir, "versions")
        self.metadata_dir = os.path.join(self.base_dir, "metadata")
        self.active_dir = os.path.join(self.base_dir, "active")
        
        for directory in [self.base_dir, self.versions_dir, self.metadata_dir, self.active_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create model type subdirectories
        self.model_types = [
            "time_sales", "order_flow", "trade_history", "sentiment",
            "option_pricing", "market_regime", "position_sizing"
        ]
        
        for model_type in self.model_types:
            os.makedirs(os.path.join(self.versions_dir, model_type), exist_ok=True)
            os.makedirs(os.path.join(self.active_dir, model_type), exist_ok=True)
        
        # Load version registry
        self.registry_file = os.path.join(self.metadata_dir, "model_registry.json")
        self.registry = self._load_registry()
        
        self.logger.info("ModelVersioning system initialized")
    
    def _load_registry(self):
        """
        Load the model version registry from file.
        
        Returns:
            dict: Model registry
        """
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading registry file: {str(e)}")
                return self._create_empty_registry()
        else:
            return self._create_empty_registry()
    
    def _create_empty_registry(self):
        """
        Create an empty model registry structure.
        
        Returns:
            dict: Empty model registry
        """
        registry = {
            "last_updated": datetime.now().isoformat(),
            "models": {}
        }
        
        for model_type in self.model_types:
            registry["models"][model_type] = {}
        
        return registry
    
    def _save_registry(self):
        """Save the model registry to file."""
        self.registry["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
            self.logger.debug("Registry saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving registry: {str(e)}")
    
    def save_model_version(self, model_obj, model_type, model_name, metadata=None, metrics=None):
        """
        Save a new version of a model.
        
        Args:
            model_obj: Model object to save
            model_type (str): Type of model (e.g., 'time_sales', 'order_flow')
            model_name (str): Name of the model
            metadata (dict, optional): Additional metadata about the model
            metrics (dict, optional): Performance metrics for this model version
            
        Returns:
            str: Version ID of the saved model
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        try:
            # Generate version ID
            version_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Create version directory
            version_dir = os.path.join(self.versions_dir, model_type, version_id)
            os.makedirs(version_dir, exist_ok=True)
            
            # Save model to file
            model_file = os.path.join(version_dir, f"{model_name}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_obj, f)
            
            # Generate model hash for integrity checking
            model_hash = self._calculate_file_hash(model_file)
            
            # Prepare metadata
            version_metadata = {
                "version_id": version_id,
                "model_name": model_name,
                "model_type": model_type,
                "created_at": timestamp,
                "file_path": model_file,
                "file_hash": model_hash,
                "metrics": metrics or {},
                "metadata": metadata or {}
            }
            
            # Add to registry
            if model_name not in self.registry["models"][model_type]:
                self.registry["models"][model_type][model_name] = {
                    "versions": [],
                    "active_version": None,
                    "created_at": timestamp,
                    "updated_at": timestamp
                }
            
            model_entry = self.registry["models"][model_type][model_name]
            model_entry["versions"].append(version_id)
            model_entry["updated_at"] = timestamp
            
            # Set as active version if first version
            if model_entry["active_version"] is None:
                model_entry["active_version"] = version_id
                self._set_active_symlink(model_type, model_name, version_id)
            
            # Save version metadata
            metadata_file = os.path.join(version_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            # Update registry
            self._save_registry()
            
            self.logger.info(f"Saved version {version_id} of {model_type} model {model_name}")
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error saving model version: {str(e)}")
            return None
    
    def _calculate_file_hash(self, file_path):
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path (str): Path to file
            
        Returns:
            str: SHA-256 hash
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        
        return sha256.hexdigest()
    
    def _set_active_symlink(self, model_type, model_name, version_id):
        """
        Set the active version symlink for a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_id (str): Version ID to set as active
        """
        active_path = os.path.join(self.active_dir, model_type, f"{model_name}.pkl")
        
        # Remove existing symlink if it exists
        if os.path.exists(active_path):
            if os.path.islink(active_path) or os.path.isfile(active_path):
                os.remove(active_path)
        
        # Get source file path
        source_path = os.path.join(self.versions_dir, model_type, version_id, f"{model_name}.pkl")
        
        # Create symlink (or copy file if symlinks not supported)
        try:
            os.symlink(source_path, active_path)
        except (OSError, AttributeError):
            # Fallback to copying if symlinks not supported
            shutil.copy2(source_path, active_path)
    
    def set_active_version(self, model_type, model_name, version_id):
        """
        Set a specific model version as active.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_id (str): Version ID to set as active
            
        Returns:
            bool: True if successful
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return False
        
        if model_name not in self.registry["models"][model_type]:
            self.logger.error(f"Model {model_name} of type {model_type} not found in registry")
            return False
        
        model_entry = self.registry["models"][model_type][model_name]
        
        if version_id not in model_entry["versions"]:
            self.logger.error(f"Version {version_id} not found for model {model_name}")
            return False
        
        try:
            # Update registry
            model_entry["active_version"] = version_id
            model_entry["updated_at"] = datetime.now().isoformat()
            
            # Update symlink
            self._set_active_symlink(model_type, model_name, version_id)
            
            # Save registry
            self._save_registry()
            
            self.logger.info(f"Set version {version_id} as active for {model_type} model {model_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting active version: {str(e)}")
            return False
    
    def get_active_version(self, model_type, model_name):
        """
        Get the active version ID for a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            
        Returns:
            str: Active version ID or None if not found
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        if model_name not in self.registry["models"][model_type]:
            self.logger.error(f"Model {model_name} of type {model_type} not found in registry")
            return None
        
        return self.registry["models"][model_type][model_name]["active_version"]
    
    def load_active_model(self, model_type, model_name):
        """
        Load the active version of a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            
        Returns:
            object: Loaded model or None if not found
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        try:
            model_path = os.path.join(self.active_dir, model_type, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                self.logger.error(f"Active model file not found for {model_type} model {model_name}")
                return None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.logger.info(f"Loaded active model for {model_type} model {model_name}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading active model: {str(e)}")
            return None
    
    def load_model_version(self, model_type, model_name, version_id):
        """
        Load a specific version of a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_id (str): Version ID to load
            
        Returns:
            object: Loaded model or None if not found
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        try:
            model_path = os.path.join(self.versions_dir, model_type, version_id, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found for version {version_id} of {model_type} model {model_name}")
                return None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.logger.info(f"Loaded version {version_id} of {model_type} model {model_name}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model version: {str(e)}")
            return None
    
    def get_model_versions(self, model_type, model_name):
        """
        Get all versions of a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            
        Returns:
            list: List of version IDs
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return []
        
        if model_name not in self.registry["models"][model_type]:
            self.logger.error(f"Model {model_name} of type {model_type} not found in registry")
            return []
        
        return self.registry["models"][model_type][model_name]["versions"]
    
    def get_version_metadata(self, model_type, model_name, version_id):
        """
        Get metadata for a specific model version.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_id (str): Version ID
            
        Returns:
            dict: Version metadata or None if not found
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        try:
            metadata_path = os.path.join(self.versions_dir, model_type, version_id, "metadata.json")
            
            if not os.path.exists(metadata_path):
                self.logger.error(f"Metadata file not found for version {version_id}")
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading version metadata: {str(e)}")
            return None
    
    def compare_versions(self, model_type, model_name, version_ids=None):
        """
        Compare multiple versions of a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_ids (list, optional): List of version IDs to compare
                                         If None, compare all versions
            
        Returns:
            pd.DataFrame: Comparison table
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        if model_name not in self.registry["models"][model_type]:
            self.logger.error(f"Model {model_name} of type {model_type} not found in registry")
            return None
        
        # Get all versions if not specified
        if version_ids is None:
            version_ids = self.registry["models"][model_type][model_name]["versions"]
        
        # Get active version
        active_version = self.registry["models"][model_type][model_name]["active_version"]
        
        # Collect metadata for each version
        version_data = []
        
        for version_id in version_ids:
            metadata = self.get_version_metadata(model_type, model_name, version_id)
            
            if metadata:
                # Extract key information
                version_info = {
                    "version_id": version_id,
                    "created_at": metadata.get("created_at", ""),
                    "is_active": version_id == active_version
                }
                
                # Add performance metrics
                metrics = metadata.get("metrics", {})
                for metric_name, metric_value in metrics.items():
                    version_info[f"metric_{metric_name}"] = metric_value
                
                # Add custom metadata
                custom_metadata = metadata.get("metadata", {})
                for meta_name, meta_value in custom_metadata.items():
                    if isinstance(meta_value, (int, float, str, bool)):
                        version_info[f"meta_{meta_name}"] = meta_value
                
                version_data.append(version_info)
        
        # Create DataFrame
        if version_data:
            comparison_df = pd.DataFrame(version_data)
            return comparison_df
        else:
            return pd.DataFrame()
    
    def delete_version(self, model_type, model_name, version_id):
        """
        Delete a specific version of a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_id (str): Version ID to delete
            
        Returns:
            bool: True if successful
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return False
        
        if model_name not in self.registry["models"][model_type]:
            self.logger.error(f"Model {model_name} of type {model_type} not found in registry")
            return False
        
        model_entry = self.registry["models"][model_type][model_name]
        
        if version_id not in model_entry["versions"]:
            self.logger.error(f"Version {version_id} not found for model {model_name}")
            return False
        
        # Cannot delete active version
        if version_id == model_entry["active_version"]:
            self.logger.error(f"Cannot delete active version {version_id}")
            return False
        
        try:
            # Remove from registry
            model_entry["versions"].remove(version_id)
            model_entry["updated_at"] = datetime.now().isoformat()
            
            # Delete files
            version_dir = os.path.join(self.versions_dir, model_type, version_id)
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            
            # Save registry
            self._save_registry()
            
            self.logger.info(f"Deleted version {version_id} of {model_type} model {model_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting version: {str(e)}")
            return False
    
    def get_model_history(self, model_type, model_name):
        """
        Get version history for a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            
        Returns:
            list: List of version metadata in chronological order
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return []
        
        if model_name not in self.registry["models"][model_type]:
            self.logger.error(f"Model {model_name} of type {model_type} not found in registry")
            return []
        
        # Get all versions
        version_ids = self.registry["models"][model_type][model_name]["versions"]
        
        # Get metadata for each version
        history = []
        
        for version_id in version_ids:
            metadata = self.get_version_metadata(model_type, model_name, version_id)
            if metadata:
                history.append(metadata)
        
        # Sort by creation date
        history.sort(key=lambda x: x.get("created_at", ""))
        
        return history
    
    def rollback_to_version(self, model_type, model_name, version_id):
        """
        Rollback to a previous version of a model.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_id (str): Version ID to rollback to
            
        Returns:
            bool: True if successful
        """
        return self.set_active_version(model_type, model_name, version_id)
    
    def export_version(self, model_type, model_name, version_id, export_dir):
        """
        Export a specific version of a model to a directory.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            version_id (str): Version ID to export
            export_dir (str): Directory to export to
            
        Returns:
            str: Path to exported files or None if failed
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        try:
            # Create export directory
            os.makedirs(export_dir, exist_ok=True)
            
            # Source paths
            source_dir = os.path.join(self.versions_dir, model_type, version_id)
            source_model = os.path.join(source_dir, f"{model_name}.pkl")
            source_metadata = os.path.join(source_dir, "metadata.json")
            
            # Target paths
            target_model = os.path.join(export_dir, f"{model_name}.pkl")
            target_metadata = os.path.join(export_dir, f"{model_name}_metadata.json")
            
            # Copy files
            if os.path.exists(source_model):
                shutil.copy2(source_model, target_model)
            else:
                self.logger.error(f"Model file not found: {source_model}")
                return None
            
            if os.path.exists(source_metadata):
                shutil.copy2(source_metadata, target_metadata)
            
            self.logger.info(f"Exported version {version_id} of {model_type} model {model_name} to {export_dir}")
            
            return export_dir
            
        except Exception as e:
            self.logger.error(f"Error exporting version: {str(e)}")
            return None
    
    def import_version(self, model_type, model_name, model_file, metadata_file=None):
        """
        Import a model from external files.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model
            model_file (str): Path to model file
            metadata_file (str, optional): Path to metadata file
            
        Returns:
            str: Version ID of imported model or None if failed
        """
        if model_type not in self.model_types:
            self.logger.error(f"Invalid model type: {model_type}")
            return None
        
        try:
            # Load model
            with open(model_file, 'rb') as f:
                model_obj = pickle.load(f)
            
            # Load metadata if available
            metadata = None
            metrics = None
            
            if metadata_file and os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata_json = json.load(f)
                    metadata = metadata_json.get("metadata", {})
                    metrics = metadata_json.get("metrics", {})
            
            # Create new version
            version_id = self.save_model_version(model_obj, model_type, model_name, metadata, metrics)
            
            self.logger.info(f"Imported {model_type} model {model_name} as version {version_id}")
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error importing model: {str(e)}")
            return None
    
    def list_models(self, model_type=None):
        """
        List all models in the registry.
        
        Args:
            model_type (str, optional): Filter by model type
            
        Returns:
            dict: Dictionary of models
        """
        result = {}
        
        if model_type is not None:
            if model_type not in self.model_types:
                self.logger.error(f"Invalid model type: {model_type}")
                return {}
            
            result[model_type] = list(self.registry["models"][model_type].keys())
        else:
            for model_type in self.model_types:
                result[model_type] = list(self.registry["models"][model_type].keys())
        
        return result
