"""
Model Registry Module

This module provides a central repository for managing ML models,
including storage, access control, and deployment management.
"""

import logging
import os
import json
import pickle
import yaml
import time
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import uuid
import sqlite3
import threading
from pathlib import Path
from ml_components.model_versioning import ModelVersioning

class ModelRegistry:
    """
    Central registry for managing ML models and their lifecycle.
    
    Features:
    - Centralized model storage and retrieval
    - Model metadata and configuration management
    - Model deployment tracking
    - Integration with versioning system
    """
    
    def __init__(self, config=None):
        """
        Initialize the ModelRegistry.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.registry_dir = "models/registry"
        self.meta_dir = os.path.join(self.registry_dir, "metadata")
        self.config_dir = os.path.join(self.registry_dir, "configs")
        self.db_dir = os.path.join(self.registry_dir, "db")
        
        for directory in [self.registry_dir, self.meta_dir, self.config_dir, self.db_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(self.db_dir, "model_registry.db")
        self._initialize_database()
        
        # Initialize versioning system
        self.versioning = ModelVersioning(config)
        
        # Cache for model objects (to avoid repeated loading)
        self.model_cache = {}
        self.cache_lock = threading.Lock()
        
        self.logger.info("ModelRegistry initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for model registry."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create models table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active_version TEXT,
                status TEXT NOT NULL
            )
            ''')
            
            # Create versions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT,
                is_active INTEGER NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
            ''')
            
            # Create metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                FOREIGN KEY (version_id) REFERENCES versions(id)
            )
            ''')
            
            # Create deployments table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS deployments (
                id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                environment TEXT NOT NULL,
                deployed_at TEXT NOT NULL,
                deployed_by TEXT,
                status TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES versions(id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def register_model(self, name, model_type, description=None):
        """
        Register a new model in the registry.
        
        Args:
            name (str): Name of the model
            model_type (str): Type of model
            description (str, optional): Description of the model
            
        Returns:
            str: Model ID
        """
        try:
            # Generate model ID
            model_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO models (id, name, type, description, created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (model_id, name, model_type, description, timestamp, timestamp, 'registered'))
            
            conn.commit()
            conn.close()
            
            # Create model directories
            os.makedirs(os.path.join(self.meta_dir, model_id), exist_ok=True)
            os.makedirs(os.path.join(self.config_dir, model_id), exist_ok=True)
            
            # Save model metadata
            metadata = {
                "id": model_id,
                "name": name,
                "type": model_type,
                "description": description,
                "created_at": timestamp,
                "updated_at": timestamp,
                "status": "registered",
                "versions": []
            }
            
            with open(os.path.join(self.meta_dir, model_id, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Registered new model: {name} ({model_id})")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            return None
    
    def add_model_version(self, model_id, model_object, config=None, metrics=None):
        """
        Add a new version of a model.
        
        Args:
            model_id (str): Model ID
            model_object: Model object to store
            config (dict, optional): Model configuration
            metrics (dict, optional): Model performance metrics
            
        Returns:
            str: Version ID
        """
        try:
            # Get model info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name, type FROM models WHERE id = ?', (model_id,))
            result = cursor.fetchone()
            
            if not result:
                self.logger.error(f"Model ID not found: {model_id}")
                conn.close()
                return None
            
            model_name, model_type = result
            
            # Get next version number
            cursor.execute('SELECT MAX(version_number) FROM versions WHERE model_id = ?', (model_id,))
            max_version = cursor.fetchone()[0]
            
            version_number = 1 if max_version is None else max_version + 1
            version_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Use versioning system to save the model
            versioning_id = self.versioning.save_model_version(
                model_object, 
                model_type, 
                model_name, 
                {"model_id": model_id, "version_number": version_number},
                metrics
            )
            
            if not versioning_id:
                self.logger.error("Failed to save model in versioning system")
                conn.close()
                return None
            
            # Get file path
            version_path = os.path.join("models/versions", model_type, versioning_id, f"{model_name}.pkl")
            
            # Insert version into database
            cursor.execute('''
            INSERT INTO versions (id, model_id, version_number, file_path, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (version_id, model_id, version_number, version_path, timestamp, 0))
            
            # Insert metrics if provided
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metric_id = str(uuid.uuid4())
                        cursor.execute('''
                        INSERT INTO metrics (id, version_id, metric_name, metric_value)
                        VALUES (?, ?, ?, ?)
                        ''', (metric_id, version_id, metric_name, metric_value))
            
            # Update model metadata
            metadata_path = os.path.join(self.meta_dir, model_id, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata["updated_at"] = timestamp
                metadata["versions"].append({
                    "id": version_id,
                    "version_number": version_number,
                    "created_at": timestamp,
                    "versioning_id": versioning_id
                })
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Save configuration if provided
            if config:
                config_path = os.path.join(self.config_dir, model_id, f"v{version_number}_config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added version {version_number} ({version_id}) to model {model_name}")
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error adding model version: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return None
    
    def activate_model_version(self, model_id, version_id):
        """
        Activate a specific version of a model.
        
        Args:
            model_id (str): Model ID
            version_id (str): Version ID to activate
            
        Returns:
            bool: True if successful
        """
        try:
            # Get model info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name, type FROM models WHERE id = ?', (model_id,))
            result = cursor.fetchone()
            
            if not result:
                self.logger.error(f"Model ID not found: {model_id}")
                conn.close()
                return False
            
            model_name, model_type = result
            
            # Check if version exists
            cursor.execute('SELECT version_number FROM versions WHERE id = ? AND model_id = ?', (version_id, model_id))
            result = cursor.fetchone()
            
            if not result:
                self.logger.error(f"Version ID not found: {version_id}")
                conn.close()
                return False
            
            version_number = result[0]
            
            # Update database
            timestamp = datetime.now().isoformat()
            
            # Reset all versions to inactive
            cursor.execute('UPDATE versions SET is_active = 0 WHERE model_id = ?', (model_id,))
            
            # Set new active version
            cursor.execute('UPDATE versions SET is_active = 1 WHERE id = ?', (version_id,))
            
            # Update model record
            cursor.execute('UPDATE models SET active_version = ?, updated_at = ? WHERE id = ?', 
                         (version_id, timestamp, model_id))
            
            # Get versioning ID for this version
            cursor.execute('''
            SELECT m.file_path FROM versions AS v
            JOIN models AS m ON v.model_id = m.id
            WHERE v.id = ?
            ''', (version_id,))
            
            result = cursor.fetchone()
            if result:
                # Extract versioning ID from file path
                file_path = result[0]
                versioning_id = file_path.split('/')[-2] if '/' in file_path else None
                
                if versioning_id:
                    # Set active version in versioning system
                    self.versioning.set_active_version(model_type, model_name, versioning_id)
            
            # Update metadata file
            metadata_path = os.path.join(self.meta_dir, model_id, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata["updated_at"] = timestamp
                metadata["active_version"] = version_id
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            conn.commit()
            conn.close()
            
            # Clear cache
            with self.cache_lock:
                if model_id in self.model_cache:
                    del self.model_cache[model_id]
            
            self.logger.info(f"Activated version {version_number} ({version_id}) of model {model_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error activating model version: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def get_model(self, model_id, version_id=None):
        """
        Get a model from the registry.
        
        Args:
            model_id (str): Model ID
            version_id (str, optional): Version ID (if None, use active version)
            
        Returns:
            object: Model object
        """
        try:
            # Check cache first
            cache_key = f"{model_id}_{version_id}" if version_id else model_id
            
            with self.cache_lock:
                if cache_key in self.model_cache:
                    self.logger.debug(f"Using cached model for {cache_key}")
                    return self.model_cache[cache_key]
            
            # Get model info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if version_id:
                # Get specific version
                cursor.execute('''
                SELECT v.file_path, m.name, m.type
                FROM versions AS v
                JOIN models AS m ON v.model_id = m.id
                WHERE v.id = ? AND v.model_id = ?
                ''', (version_id, model_id))
            else:
                # Get active version
                cursor.execute('''
                SELECT v.file_path, m.name, m.type
                FROM versions AS v
                JOIN models AS m ON v.model_id = m.id
                WHERE v.model_id = ? AND v.is_active = 1
                ''', (model_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                self.logger.error(f"Model version not found for model {model_id}")
                return None
            
            file_path, model_name, model_type = result
            
            # Extract versioning ID from file path
            versioning_id = file_path.split('/')[-2] if '/' in file_path else None
            
            if not versioning_id:
                self.logger.error(f"Invalid file path format: {file_path}")
                return None
            
            # Use versioning system to load the model
            model = self.versioning.load_model_version(model_type, model_name, versioning_id)
            
            if model:
                # Cache the loaded model
                with self.cache_lock:
                    self.model_cache[cache_key] = model
                
                self.logger.info(f"Loaded model {model_name} ({model_id}), version {version_id or 'active'}")
                return model
            else:
                self.logger.error(f"Failed to load model from {file_path}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting model: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return None
    
    def get_model_metadata(self, model_id):
        """
        Get metadata for a model.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            dict: Model metadata
        """
        try:
            metadata_path = os.path.join(self.meta_dir, model_id, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                return metadata
            
            # Fallback to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, name, type, description, created_at, updated_at, active_version, status
            FROM models WHERE id = ?
            ''', (model_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                self.logger.error(f"Model ID not found: {model_id}")
                return None
            
            metadata = {
                "id": result[0],
                "name": result[1],
                "type": result[2],
                "description": result[3],
                "created_at": result[4],
                "updated_at": result[5],
                "active_version": result[6],
                "status": result[7],
                "versions": []
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting model metadata: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return None
    
    def get_version_metrics(self, version_id):
        """
        Get performance metrics for a model version.
        
        Args:
            version_id (str): Version ID
            
        Returns:
            dict: Performance metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT metric_name, metric_value FROM metrics
            WHERE version_id = ?
            ''', (version_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {}
            
            metrics = {row[0]: row[1] for row in results}
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting version metrics: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return {}
    
    def list_models(self, model_type=None):
        """
        List all models in the registry.
        
        Args:
            model_type (str, optional): Filter by model type
            
        Returns:
            list: List of model records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if model_type:
                cursor.execute('''
                SELECT id, name, type, description, created_at, updated_at, active_version, status
                FROM models WHERE type = ?
                ''', (model_type,))
            else:
                cursor.execute('''
                SELECT id, name, type, description, created_at, updated_at, active_version, status
                FROM models
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            models = []
            for row in results:
                model = {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "active_version": row[6],
                    "status": row[7]
                }
                models.append(model)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return []
    
    def list_model_versions(self, model_id):
        """
        List all versions of a model.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            list: List of version records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, version_number, file_path, created_at, is_active
            FROM versions WHERE model_id = ?
            ORDER BY version_number
            ''', (model_id,))
            
            results = cursor.fetchall()
            
            versions = []
            for row in results:
                version_id = row[0]
                
                # Get metrics for this version
                cursor.execute('''
                SELECT metric_name, metric_value FROM metrics
                WHERE version_id = ?
                ''', (version_id,))
                
                metric_results = cursor.fetchall()
                metrics = {m[0]: m[1] for m in metric_results}
                
                version = {
                    "id": version_id,
                    "version_number": row[1],
                    "file_path": row[2],
                    "created_at": row[3],
                    "is_active": bool(row[4]),
                    "metrics": metrics
                }
                versions.append(version)
            
            conn.close()
            return versions
            
        except Exception as e:
            self.logger.error(f"Error listing model versions: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return []
    
    def deploy_model(self, model_id, version_id, environment, deployed_by=None):
        """
        Record a model deployment.
        
        Args:
            model_id (str): Model ID
            version_id (str): Version ID
            environment (str): Deployment environment
            deployed_by (str, optional): User who deployed the model
            
        Returns:
            str: Deployment ID
        """
        try:
            # Check if model and version exist
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM models WHERE id = ?', (model_id,))
            if not cursor.fetchone():
                self.logger.error(f"Model ID not found: {model_id}")
                conn.close()
                return None
            
            cursor.execute('SELECT id FROM versions WHERE id = ? AND model_id = ?', (version_id, model_id))
            if not cursor.fetchone():
                self.logger.error(f"Version ID not found: {version_id}")
                conn.close()
                return None
            
            # Record deployment
            deployment_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO deployments (id, version_id, environment, deployed_at, deployed_by, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (deployment_id, version_id, environment, timestamp, deployed_by, 'deployed'))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Recorded deployment {deployment_id} of version {version_id} to {environment}")
            
            return deployment_id
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return None
    
    def get_model_deployments(self, model_id):
        """
        Get deployment history for a model.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            list: List of deployment records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT d.id, d.version_id, v.version_number, d.environment, d.deployed_at, d.deployed_by, d.status
            FROM deployments AS d
            JOIN versions AS v ON d.version_id = v.id
            WHERE v.model_id = ?
            ORDER BY d.deployed_at DESC
            ''', (model_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            deployments = []
            for row in results:
                deployment = {
                    "id": row[0],
                    "version_id": row[1],
                    "version_number": row[2],
                    "environment": row[3],
                    "deployed_at": row[4],
                    "deployed_by": row[5],
                    "status": row[6]
                }
                deployments.append(deployment)
            
            return deployments
            
        except Exception as e:
            self.logger.error(f"Error getting model deployments: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return []
    
    def compare_model_versions(self, model_id, version_ids=None):
        """
        Compare multiple versions of a model.
        
        Args:
            model_id (str): Model ID
            version_ids (list, optional): List of version IDs to compare
                                         If None, compare all versions
            
        Returns:
            pd.DataFrame: Comparison table
        """
        try:
            # Get all versions if not specified
            if version_ids is None:
                versions = self.list_model_versions(model_id)
                version_ids = [v["id"] for v in versions]
            
            # Collect metrics for each version
            version_data = []
            
            for version_id in version_ids:
                # Get version info
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT version_number, created_at, is_active
                FROM versions WHERE id = ? AND model_id = ?
                ''', (version_id, model_id))
                
                result = cursor.fetchone()
                
                if not result:
                    conn.close()
                    continue
                
                version_number, created_at, is_active = result
                
                # Get metrics
                metrics = self.get_version_metrics(version_id)
                
                # Create version record
                version_record = {
                    "version_id": version_id,
                    "version_number": version_number,
                    "created_at": created_at,
                    "is_active": bool(is_active)
                }
                
                # Add metrics
                for metric_name, metric_value in metrics.items():
                    version_record[f"metric_{metric_name}"] = metric_value
                
                version_data.append(version_record)
                
                conn.close()
            
            # Create DataFrame
            if version_data:
                comparison_df = pd.DataFrame(version_data)
                return comparison_df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error comparing model versions: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return pd.DataFrame()
    
    def archive_model(self, model_id):
        """
        Archive a model (mark as inactive).
        
        Args:
            model_id (str): Model ID
            
        Returns:
            bool: True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE models SET status = ? WHERE id = ?', ('archived', model_id))
            
            if cursor.rowcount == 0:
                self.logger.error(f"Model ID not found: {model_id}")
                conn.close()
                return False
            
            timestamp = datetime.now().isoformat()
            
            # Update metadata file
            metadata_path = os.path.join(self.meta_dir, model_id, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata["updated_at"] = timestamp
                metadata["status"] = "archived"
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            conn.commit()
            conn.close()
            
            # Clear cache
            with self.cache_lock:
                keys_to_remove = [k for k in self.model_cache if k.startswith(model_id)]
                for key in keys_to_remove:
                    del self.model_cache[key]
            
            self.logger.info(f"Archived model {model_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error archiving model: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def search_models(self, query=None, model_type=None, status=None):
        """
        Search for models in the registry.
        
        Args:
            query (str, optional): Search query for name or description
            model_type (str, optional): Filter by model type
            status (str, optional): Filter by status
            
        Returns:
            list: List of matching model records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql = '''
            SELECT id, name, type, description, created_at, updated_at, active_version, status
            FROM models WHERE 1=1
            '''
            
            params = []
            
            if query:
                sql += ' AND (name LIKE ? OR description LIKE ?)'
                params.extend([f'%{query}%', f'%{query}%'])
            
            if model_type:
                sql += ' AND type = ?'
                params.append(model_type)
            
            if status:
                sql += ' AND status = ?'
                params.append(status)
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
            conn.close()
            
            models = []
            for row in results:
                model = {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "active_version": row[6],
                    "status": row[7]
                }
                models.append(model)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error searching models: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return []
    
    def export_model(self, model_id, version_id=None, export_dir=None):
        """
        Export a model and its metadata.
        
        Args:
            model_id (str): Model ID
            version_id (str, optional): Version ID (if None, use active version)
            export_dir (str, optional): Directory to export to
            
        Returns:
            str: Path to exported files or None if failed
        """
        if export_dir is None:
            export_dir = os.path.join("exports", f"model_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        try:
            # Get model info
            model_metadata = self.get_model_metadata(model_id)
            
            if not model_metadata:
                self.logger.error(f"Model ID not found: {model_id}")
                return None
            
            model_name = model_metadata.get("name")
            model_type = model_metadata.get("type")
            
            # Get version ID if not specified
            if not version_id:
                version_id = model_metadata.get("active_version")
                
                if not version_id:
                    self.logger.error(f"No active version found for model {model_id}")
                    return None
            
            # Create export directory
            os.makedirs(export_dir, exist_ok=True)
            
            # Get version info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT version_number, file_path FROM versions WHERE id = ?', (version_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                self.logger.error(f"Version ID not found: {version_id}")
                return None
            
            version_number, file_path = result
            
            # Extract versioning ID from file path
            versioning_id = file_path.split('/')[-2] if '/' in file_path else None
            
            if not versioning_id:
                self.logger.error(f"Invalid file path format: {file_path}")
                return None
            
            # Use versioning system to export
            export_path = self.versioning.export_version(model_type, model_name, versioning_id, export_dir)
            
            if not export_path:
                self.logger.error(f"Failed to export model version")
                return None
            
            # Export metadata
            with open(os.path.join(export_dir, "model_metadata.json"), 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Export metrics
            metrics = self.get_version_metrics(version_id)
            with open(os.path.join(export_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Exported model {model_name} (version {version_number}) to {export_dir}")
            
            return export_dir
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            return None
