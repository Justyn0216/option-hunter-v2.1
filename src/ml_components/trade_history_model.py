"""
Trade History Model Module

This module implements machine learning models for analyzing historical trade data
to identify successful patterns, optimize parameters, and improve future trading decisions.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class TradeHistoryModel:
    """
    Machine learning model for analyzing historical trades to improve future performance.
    
    Features:
    - Trade success prediction
    - Exit timing optimization
    - Parameter optimization based on historical results
    - Trade clustering and pattern identification
    """
    
    def __init__(self, config=None):
        """
        Initialize the TradeHistoryModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = "models/trade_history"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.model_type = self.config.get('trade_history_model_type', 'xgboost')
        self.prediction_type = self.config.get('trade_history_prediction_type', 'classification')
        
        # Initialize model
        self.model = None
        self.preprocessor = None
        self.feature_columns = []
        
        self.logger.info(f"TradeHistoryModel initialized with {self.model_type} model for {self.prediction_type} task")
    
    def preprocess_data(self, trade_data):
        """
        Preprocess trade history data for model training.
        
        Args:
            trade_data (pd.DataFrame or list): Historical trade data
            
        Returns:
            tuple: X (features) and y (targets) for model training
        """
        self.logger.info("Preprocessing trade history data")
        
        # Convert list of dictionaries to DataFrame if needed
        if isinstance(trade_data, list):
            df = pd.DataFrame(trade_data)
        else:
            df = trade_data.copy()
        
        # Check if we have the required columns
        required_columns = ['symbol', 'entry_price', 'exit_price', 'entry_date', 'exit_date', 'pnl']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            self.logger.error(f"Missing required columns: {missing}")
            return None, None
        
        # Convert date columns to datetime
        for date_col in ['entry_date', 'exit_date']:
            if not pd.api.types.is_datetime64_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
        
        # Feature engineering
        
        # Calculate trade duration in days
        df['duration'] = (df['exit_date'] - df['entry_date']).dt.total_seconds() / (60 * 60 * 24)
        
        # Calculate return percentage
        if 'pnl_percent' not in df.columns:
            if 'side' in df.columns:
                # Handle both long and short trades
                df['pnl_percent'] = np.where(
                    df['side'] == 'long',
                    (df['exit_price'] - df['entry_price']) / df['entry_price'],
                    (df['entry_price'] - df['exit_price']) / df['entry_price']
                )
            else:
                # Assume all trades are long
                df['pnl_percent'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
        
        # Extract temporal features
        df['entry_day_of_week'] = df['entry_date'].dt.dayofweek
        df['entry_hour'] = df['entry_date'].dt.hour
        df['entry_month'] = df['entry_date'].dt.month
        df['exit_day_of_week'] = df['exit_date'].dt.dayofweek
        df['exit_hour'] = df['exit_date'].dt.hour
        
        # Create success flag (1 for profitable trades, 0 for losing trades)
        df['success'] = (df['pnl'] > 0).astype(int)
        
        # Extract option-specific features if available
        if 'option_type' in df.columns:
            df['is_call'] = (df['option_type'] == 'call').astype(int)
        
        if 'strike' in df.columns and 'underlying_price' in df.columns:
            df['moneyness'] = np.where(
                df.get('option_type', 'call') == 'call',
                df['underlying_price'] / df['strike'] - 1,
                df['strike'] / df['underlying_price'] - 1
            )
        
        # Extract Greeks if available
        greek_columns = []
        if 'greeks' in df.columns:
            # Handle if greeks are stored as dictionaries
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                if any(greek in x for x in df['greeks'] if isinstance(x, dict)):
                    df[greek] = df['greeks'].apply(lambda x: x.get(greek, 0) if isinstance(x, dict) else 0)
                    greek_columns.append(greek)
        else:
            # Check if individual greek columns exist
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                if greek in df.columns:
                    greek_columns.append(greek)
        
        # Handle days to expiration
        if 'expiration' in df.columns:
            if not pd.api.types.is_datetime64_dtype(df['expiration']):
                df['expiration'] = pd.to_datetime(df['expiration'])
            
            df['days_to_expiration_at_entry'] = (df['expiration'] - df['entry_date']).dt.total_seconds() / (60 * 60 * 24)
            df['days_to_expiration_at_exit'] = (df['expiration'] - df['exit_date']).dt.total_seconds() / (60 * 60 * 24)
        
        # Add market regime if available
        if 'market_regime' in df.columns:
            df['market_bullish'] = (df['market_regime'] == 'bullish').astype(int)
            df['market_bearish'] = (df['market_regime'] == 'bearish').astype(int)
            df['market_neutral'] = ((df['market_regime'] != 'bullish') & (df['market_regime'] != 'bearish')).astype(int)
        
        # Add volatility regime if available
        if 'volatility_regime' in df.columns:
            df['vol_high'] = (df['volatility_regime'] == 'high').astype(int)
            df['vol_low'] = (df['volatility_regime'] == 'low').astype(int)
        
        # Add sentiment data if available
        if 'sentiment' in df.columns:
            sentiment_map = {
                'very_bullish': 2,
                'bullish': 1,
                'neutral': 0,
                'bearish': -1,
                'very_bearish': -2
            }
            
            df['sentiment_score'] = df['sentiment'].map(lambda x: sentiment_map.get(x, 0))
        
        # Determine trade exit reason code if available
        if 'exit_reason' in df.columns:
            # One-hot encode exit reasons
            exit_reasons = pd.get_dummies(df['exit_reason'], prefix='exit_reason')
            df = pd.concat([df, exit_reasons], axis=1)
        
        # Select features based on prediction type
        numeric_features = ['entry_price', 'duration', 'entry_day_of_week', 
                          'entry_hour', 'entry_month']
        
        # Add option-specific features if available
        if 'is_call' in df.columns:
            numeric_features.append('is_call')
        
        if 'moneyness' in df.columns:
            numeric_features.append('moneyness')
        
        if 'days_to_expiration_at_entry' in df.columns:
            numeric_features.append('days_to_expiration_at_entry')
        
        # Add Greek columns if available
        numeric_features.extend([col for col in greek_columns if col in df.columns])
        
        # Add market and volatility regime features if available
        regime_features = ['market_bullish', 'market_bearish', 'market_neutral',
                         'vol_high', 'vol_low', 'sentiment_score']
        
        numeric_features.extend([col for col in regime_features if col in df.columns])
        
        # Add additional features if available
        if 'entry_score' in df.columns:
            numeric_features.append('entry_score')
        
        if 'quantity' in df.columns:
            numeric_features.append('quantity')
        
        # Select categorical features if any
        categorical_features = []  # e.g., symbol if we want to encode it
        
        # Define target variable based on prediction type
        if self.prediction_type == 'classification':
            # Binary classification: predict trade success
            target = df['success']
        else:
            # Regression: predict return percentage
            target = df['pnl_percent']
        
        # Final feature selection
        self.feature_columns = numeric_features + categorical_features
        
        # Ensure we have features and target
        if not self.feature_columns:
            self.logger.error("No valid features found for model training")
            return None, None
        
        # Extract features and ensure all are numeric
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Return features and target
        self.logger.info(f"Preprocessed data: {X.shape[0]} trades with {X.shape[1]} features")
        
        return X, target
    
    def build_preprocessor(self, X):
        """
        Build a preprocessor for feature transformation.
        
        Args:
            X (pd.DataFrame): Feature DataFrame
            
        Returns:
            ColumnTransformer: Scikit-learn preprocessor
        """
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        # Add numeric transformer
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        # Add categorical transformer
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        return preprocessor
    
    def build_model(self, model_type=None, prediction_type=None):
        """
        Build the machine learning model.
        
        Args:
            model_type (str, optional): Type of model to build
            prediction_type (str, optional): Type of prediction task (classification or regression)
            
        Returns:
            object: Model instance
        """
        if model_type is None:
            model_type = self.model_type
            
        if prediction_type is None:
            prediction_type = self.prediction_type
        
        if prediction_type == 'classification':
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
                
            elif model_type == 'xgboost':
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    scale_pos_weight=1,  # Adjust based on class imbalance
                    random_state=42,
                    n_jobs=-1
                )
                
            elif model_type == 'lightgbm':
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
            else:
                raise ValueError(f"Unsupported classification model type: {model_type}")
                
        else:  # regression
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
                
            elif model_type == 'xgboost':
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1
                )
                
            elif model_type == 'lightgbm':
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                
            else:
                raise ValueError(f"Unsupported regression model type: {model_type}")
        
        return model
    
    def train(self, trade_data, model_name=None, test_size=0.2, optimize_hyperparams=False):
        """
        Train the trade history model.
        
        Args:
            trade_data (pd.DataFrame or list): Historical trade data
            model_name (str, optional): Name for the saved model
            test_size (float): Proportion of data to use for testing
            optimize_hyperparams (bool): Whether to perform hyperparameter optimization
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training {self.model_type} model on trade history data")
        
        # Preprocess data
        X, y = self.preprocess_data(trade_data)
        
        if X is None or y is None:
            self.logger.error("Preprocessing failed, cannot train model")
            return None
        
        # Build preprocessor
        self.preprocessor = self.build_preprocessor(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Preprocess features
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Build model
        if optimize_hyperparams:
            self.model = self._optimize_hyperparameters(X_train_transformed, y_train)
        else:
            self.model = self.build_model()
        
        # Train model
        self.model.fit(X_train_transformed, y_train)
        
        # Evaluate model
        metrics = self._evaluate_model(X_test_transformed, y_test)
        
        # Save model
        if model_name:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'preprocessor': self.preprocessor,
                    'feature_columns': self.feature_columns,
                    'config': {
                        'model_type': self.model_type,
                        'prediction_type': self.prediction_type
                    }
                }, f)
            
            self.logger.info(f"Model saved to {model_path}")
        
        return metrics
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize model hyperparameters using grid search.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            
        Returns:
            object: Optimized model
        """
        self.logger.info("Optimizing model hyperparameters")
        
        if self.prediction_type == 'classification':
            if self.model_type == 'random_forest':
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [5, 10, 15],
                    'min_samples_leaf': [2, 5, 10]
                }
                
            elif self.model_type == 'xgboost':
                model = XGBClassifier(random_state=42, n_jobs=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
                
            elif self.model_type == 'lightgbm':
                model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.6, 0.8, 1.0]
                }
        else:  # regression
            if self.model_type == 'random_forest':
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [5, 10, 15],
                    'min_samples_leaf': [2, 5, 10]
                }
                
            elif self.model_type == 'xgboost':
                model = XGBRegressor(random_state=42, n_jobs=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, the code stops here, but expected to continue with more parameters
