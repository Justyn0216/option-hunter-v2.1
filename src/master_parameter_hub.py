"""
Master Parameter Hub

This module serves as the central parameter control system,
automatically adjusting trading parameters based on historical performance
and machine learning insights.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class MasterParameterHub:
    """
    Central parameter management system that adapts trading parameters
    based on historical performance and market conditions.
    """
    
    def __init__(self, config, drive_manager, disable_ml=False):
        """
        Initialize the MasterParameterHub.
        
        Args:
            config (dict): Main configuration dictionary
            drive_manager: Google Drive manager for data storage and retrieval
            disable_ml (bool): If True, disables ML-based parameter adjustment
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.drive_manager = drive_manager
        self.disable_ml = disable_ml
        
        # Default parameters (from config)
        self.trading_params = config["trade_parameters"].copy()
        
        # ML models and data
        self.ml_models = {}
        self.parameter_history = []
        self.last_training_date = None
        
        # Initialize parameter directory
        self.param_dir = "data/parameters"
        os.makedirs(self.param_dir, exist_ok=True)
        
        # Load parameter history if available
        self._load_parameter_history()
        
        # Initialize ML models if enabled
        if not disable_ml:
            self._initialize_ml_models()
        else:
            self.logger.info("ML-based parameter adjustment disabled")
    
    def _load_parameter_history(self):
        """Load parameter history from local file or Google Drive."""
        try:
            # Try local file first
            history_file = f"{self.param_dir}/parameter_history.json"
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    self.parameter_history = json.load(f)
                self.logger.info(f"Loaded parameter history with {len(self.parameter_history)} entries")
            else:
                # Try Google Drive
                history_drive_file = "parameter_history.json"
                if self.drive_manager.file_exists(history_drive_file):
                    content = self.drive_manager.download_file(history_drive_file)
                    self.parameter_history = json.loads(content)
                    self.logger.info(f"Loaded parameter history from Google Drive with {len(self.parameter_history)} entries")
                    
                    # Save locally for future use
                    with open(history_file, "w") as f:
                        json.dump(self.parameter_history, f, indent=2)
                else:
                    self.logger.info("No parameter history found. Using default parameters.")
        except Exception as e:
            self.logger.error(f"Error loading parameter history: {str(e)}")
            self.logger.info("Using default parameters due to error.")
    
    def _initialize_ml_models(self):
        """Initialize ML models for parameter adjustment."""
        try:
            # Look for saved models
            model_file = f"{self.param_dir}/parameter_models.joblib"
            if os.path.exists(model_file):
                self.ml_models = joblib.load(model_file)
                self.logger.info(f"Loaded ML models for parameters: {list(self.ml_models.keys())}")
            else:
                # Try Google Drive
                model_drive_file = "parameter_models.joblib"
                if self.drive_manager.file_exists(model_drive_file):
                    content = self.drive_manager.download_file_binary(model_drive_file)
                    # Save locally
                    with open(model_file, "wb") as f:
                        f.write(content)
                    self.ml_models = joblib.load(model_file)
                    self.logger.info(f"Loaded ML models from Google Drive for parameters: {list(self.ml_models.keys())}")
                else:
                    # Create new models
                    self._create_initial_models()
            
            # Check if we need to train the models with the latest data
            self._check_retrain_models()
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {str(e)}")
            self.logger.info("Using default parameters due to error.")
    
    def _create_initial_models(self):
        """Create initial ML models for parameter adjustment."""
        parameters_to_optimize = [
            "max_risk_per_trade_percentage",
            "default_stop_loss_percentage",
            "take_profit_percentage",
            "max_position_size_per_symbol_percentage"
        ]
        
        for param in parameters_to_optimize:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            self.ml_models[param] = model
        
        self.logger.info(f"Created initial ML models for parameters: {parameters_to_optimize}")
    
    def _check_retrain_models(self):
        """Check if models need to be retrained based on new data."""
        if not self.parameter_history or len(self.parameter_history) < 30:
            self.logger.info("Not enough parameter history data for retraining")
            return
        
        # Check if we've trained recently
        today = datetime.now().date()
        train_frequency = timedelta(days=self.config["ml_parameters"]["training_frequency_days"])
        
        if (self.last_training_date is None or 
            (today - self.last_training_date) >= train_frequency):
            self.logger.info("Retraining parameter models with latest data")
            self._train_models()
            self.last_training_date = today
        else:
            self.logger.debug("No retraining needed. Last training was recent.")
    
    def _train_models(self):
        """Train the ML models using historical parameter data."""
        try:
            # Convert parameter history to DataFrame
            history_df = pd.DataFrame(self.parameter_history)
            
            # Ensure we have required columns
            required_columns = ['date', 'market_conditions', 'winning_trades', 
                                'losing_trades', 'profit_factor', 'sharpe_ratio']
            
            if not all(col in history_df.columns for col in required_columns):
                self.logger.error("Parameter history missing required columns. Cannot train models.")
                return
            
            # Create feature columns (market conditions, win rate, etc.)
            history_df['win_rate'] = history_df['winning_trades'] / (history_df['winning_trades'] + history_df['losing_trades'])
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df['day_of_week'] = history_df['date'].dt.dayofweek
            
            # Create dummy variables for market_conditions
            market_dummies = pd.get_dummies(history_df['market_conditions'], prefix='market')
            history_df = pd.concat([history_df, market_dummies], axis=1)
            
            # Define features
            feature_columns = [col for col in history_df.columns if col.startswith('market_')] + [
                'win_rate', 'profit_factor', 'sharpe_ratio', 'day_of_week'
            ]
            
            # Train a model for each parameter
            for param, model in self.ml_models.items():
                if param in history_df.columns:
                    X = history_df[feature_columns]
                    y = history_df[param]
                    model.fit(X, y)
                    self.logger.info(f"Trained model for parameter: {param}")
            
            # Save the trained models
            model_file = f"{self.param_dir}/parameter_models.joblib"
            joblib.dump(self.ml_models, model_file)
            
            # Upload to Google Drive
            with open(model_file, 'rb') as f:
                self.drive_manager.upload_file('parameter_models.joblib', f.read(), mime_type='application/octet-stream')
                
            self.logger.info("Parameter models trained and saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error training parameter models: {str(e)}")
    
    def get_optimized_parameters(self, market_conditions, current_performance=None):
        """
        Get optimized trading parameters based on market conditions and current performance.
        
        Args:
            market_conditions (str): Current market conditions/regime
            current_performance (dict, optional): Current trading performance metrics
            
        Returns:
            dict: Optimized trading parameters
        """
        # If ML is disabled or no models available, return default parameters
        if self.disable_ml or not self.ml_models:
            return self.trading_params
        
        try:
            # Prepare features for prediction
            features = {}
            
            # Market condition encoding (one-hot)
            market_conditions_list = ['bullish', 'bearish', 'sideways', 'volatile', 'trending']
            for market in market_conditions_list:
                features[f'market_{market}'] = 1 if market_conditions == market else 0
            
            # Performance metrics
            if current_performance:
                # Calculate win rate
                total_trades = current_performance.get('winning_trades', 0) + current_performance.get('losing_trades', 0)
                win_rate = current_performance.get('winning_trades', 0) / total_trades if total_trades > 0 else 0.5
                
                features['win_rate'] = win_rate
                features['profit_factor'] = current_performance.get('profit_factor', 1.5)
                features['sharpe_ratio'] = current_performance.get('sharpe_ratio', 1.0)
            else:
                # Default values if no performance data available
                features['win_rate'] = 0.5
                features['profit_factor'] = 1.5
                features['sharpe_ratio'] = 1.0
            
            # Day of week
            features['day_of_week'] = datetime.now().weekday()
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([features])
            
            # Predict optimized parameters
            optimized_params = self.trading_params.copy()
            for param, model in self.ml_models.items():
                if param in self.trading_params:
                    try:
                        predicted_value = model.predict(features_df)[0]
                        
                        # Apply reasonable bounds based on parameter type
                        if param.endswith('percentage'):
                            # Percentage parameters should be positive and reasonable
                            predicted_value = max(0.1, min(100.0, predicted_value))
                        
                        optimized_params[param] = predicted_value
                        self.logger.debug(f"Optimized {param}: {self.trading_params[param]} -> {predicted_value}")
                    except Exception as e:
                        self.logger.error(f"Error predicting {param}: {str(e)}")
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            return self.trading_params
    
    def update_parameter_history(self, parameters, performance_metrics, market_conditions):
        """
        Update parameter history with new data.
        
        Args:
            parameters (dict): The trading parameters used
            performance_metrics (dict): Performance metrics achieved with these parameters
            market_conditions (str): Market conditions during this period
        """
        # Create entry
        entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'market_conditions': market_conditions,
            **parameters,
            **performance_metrics
        }
        
        # Add to history
        self.parameter_history.append(entry)
        
        # Save history
        try:
            history_file = f"{self.param_dir}/parameter_history.json"
            with open(history_file, "w") as f:
                json.dump(self.parameter_history, f, indent=2)
            
            # Upload to Google Drive
            self.drive_manager.upload_file(
                'parameter_history.json', 
                json.dumps(self.parameter_history, indent=2),
                mime_type='application/json'
            )
            
            self.logger.info(f"Updated parameter history with entry from {entry['date']}")
            
            # Check if we should retrain models
            self._check_retrain_models()
            
        except Exception as e:
            self.logger.error(f"Error saving parameter history: {str(e)}")
    
    def get_parameter(self, param_name, default=None):
        """
        Get a specific parameter value.
        
        Args:
            param_name (str): Name of the parameter
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default if not found
        """
        return self.trading_params.get(param_name, default)
    
    def set_parameter(self, param_name, value):
        """
        Manually set a parameter value.
        
        Args:
            param_name (str): Name of the parameter
            value: New value for the parameter
        """
        self.trading_params[param_name] = value
        self.logger.info(f"Manually set {param_name} = {value}")
    
    def reset_to_defaults(self):
        """Reset all parameters to their default values from config."""
        self.trading_params = self.config["trade_parameters"].copy()
        self.logger.info("Reset all parameters to default values")
    
    def get_all_parameters(self):
        """
        Get all current trading parameters.
        
        Returns:
            dict: All current trading parameters
        """
        return self.trading_params.copy()
