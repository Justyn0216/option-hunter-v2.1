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
'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
                
            elif self.model_type == 'lightgbm':
                model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.6, 0.8, 1.0]
                }
        
        # Reduce grid size if we have limited data
        if len(y_train) < 100:
            for param in param_grid:
                if isinstance(param_grid[param], list) and len(param_grid[param]) > 2:
                    param_grid[param] = [param_grid[param][0], param_grid[param][-1]]
        
        # Set up grid search
        if self.prediction_type == 'classification':
            scoring = 'f1'
        else:
            scoring = 'neg_mean_squared_error'
        
        # Reduce CV folds for smaller datasets
        cv = min(5, max(2, len(y_train) // 20))
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {}
        
        if self.prediction_type == 'classification':
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)[:, 1]
                metrics['probability_threshold'] = 0.5
            else:
                y_prob = None
            
            # Calculate metrics
            metrics['accuracy'] = np.mean(y_pred == y_test)
            
            # Calculate precision, recall, F1
            from sklearn.metrics import precision_recall_fscore_support
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            metrics['precision'] = p
            metrics['recall'] = r
            metrics['f1_score'] = f1
            
            # Calculate ROC AUC if probabilities available
            if y_prob is not None:
                from sklearn.metrics import roc_auc_score
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            
            # Detailed classification report
            from sklearn.metrics import classification_report
            metrics['classification_report'] = classification_report(y_test, y_pred)
            
        else:  # regression
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importances'] = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        return metrics
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to saved model
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model = saved_data['model']
            self.preprocessor = saved_data['preprocessor']
            self.feature_columns = saved_data.get('feature_columns', [])
            
            # Load configuration
            if 'config' in saved_data:
                config = saved_data['config']
                self.model_type = config.get('model_type', self.model_type)
                self.prediction_type = config.get('prediction_type', self.prediction_type)
            
            self.logger.info(f"Loaded {self.model_type} model from {model_path}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def predict(self, trade_data):
        """
        Make predictions on new trade data.
        
        Args:
            trade_data (pd.DataFrame or dict): Trade data
            
        Returns:
            dict: Prediction results
        """
        if self.model is None or self.preprocessor is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        self.logger.info("Making predictions on trade data")
        
        try:
            # Convert single trade dict to DataFrame
            if isinstance(trade_data, dict):
                trade_data = pd.DataFrame([trade_data])
            
            # Preprocess data
            X, _ = self.preprocess_data(trade_data)
            
            if X is None:
                self.logger.error("Preprocessing failed, cannot make predictions")
                return None
            
            # Ensure all required features are present
            missing_features = [col for col in self.feature_columns if col not in X.columns]
            
            # Fill missing features with zeros
            for feature in missing_features:
                X[feature] = 0
            
            # Reorder columns to match training data
            X = X[self.feature_columns]
            
            # Transform features
            X_transformed = self.preprocessor.transform(X)
            
            # Make predictions
            if self.prediction_type == 'classification':
                # Predict success probability
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X_transformed)[:, 1]
                else:
                    probabilities = self.model.predict(X_transformed).astype(float)
                
                predictions = (probabilities >= 0.5).astype(int)
                
                # Prepare results
                results = []
                
                for i, (_, row) in enumerate(trade_data.iterrows()):
                    if i >= len(predictions):
                        continue
                        
                    result = {
                        'trade_id': row.get('id', f"trade_{i}"),
                        'symbol': row.get('symbol', 'unknown'),
                        'prediction': int(predictions[i]),
                        'success_probability': float(probabilities[i]),
                        'expected_outcome': 'success' if predictions[i] == 1 else 'failure'
                    }
                    
                    # Add confidence level
                    confidence = abs(probabilities[i] - 0.5) * 2  # Scale to 0-1
                    result['confidence'] = float(confidence)
                    
                    results.append(result)
                
            else:  # regression
                # Predict return percentage
                predictions = self.model.predict(X_transformed)
                
                # Prepare results
                results = []
                
                for i, (_, row) in enumerate(trade_data.iterrows()):
                    if i >= len(predictions):
                        continue
                        
                    result = {
                        'trade_id': row.get('id', f"trade_{i}"),
                        'symbol': row.get('symbol', 'unknown'),
                        'predicted_return': float(predictions[i]),
                        'expected_outcome': 'success' if predictions[i] > 0 else 'failure'
                    }
                    
                    results.append(result)
            
            self.logger.info(f"Generated predictions for {len(results)} trades")
            
            return {
                'results': results,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def cluster_trades(self, trade_data, n_clusters=4):
        """
        Cluster trades into groups based on features.
        
        Args:
            trade_data (pd.DataFrame): Trade data
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Clustering results
        """
        self.logger.info(f"Clustering trades into {n_clusters} groups")
        
        try:
            # Preprocess data
            X, _ = self.preprocess_data(trade_data)
            
            if X is None:
                self.logger.error("Preprocessing failed, cannot cluster trades")
                return None
            
            # Build preprocessor if not already built
            if self.preprocessor is None:
                self.preprocessor = self.build_preprocessor(X)
            
            # Transform features
            X_transformed = self.preprocessor.fit_transform(X)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_transformed)
            
            # Add cluster labels to the original data
            trade_data = trade_data.copy()
            trade_data['cluster'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = {}
            
            for cluster_id in range(n_clusters):
                cluster_trades = trade_data[trade_data['cluster'] == cluster_id]
                
                # Trade success rate
                success_rate = (cluster_trades['pnl'] > 0).mean() if 'pnl' in cluster_trades.columns else None
                
                # Average PnL
                avg_pnl = cluster_trades['pnl'].mean() if 'pnl' in cluster_trades.columns else None
                
                # Average duration
                avg_duration = None
                if all(col in cluster_trades.columns for col in ['entry_date', 'exit_date']):
                    durations = (cluster_trades['exit_date'] - cluster_trades['entry_date']).dt.total_seconds() / (60 * 60 * 24)
                    avg_duration = durations.mean()
                
                # Most common symbols
                common_symbols = None
                if 'symbol' in cluster_trades.columns:
                    symbol_counts = cluster_trades['symbol'].value_counts()
                    common_symbols = symbol_counts.to_dict()
                
                # Store statistics
                cluster_stats[cluster_id] = {
                    'count': len(cluster_trades),
                    'success_rate': success_rate,
                    'avg_pnl': avg_pnl,
                    'avg_duration': avg_duration,
                    'common_symbols': common_symbols,
                    'sample_trades': cluster_trades.head(5).to_dict('records') if len(cluster_trades) > 0 else []
                }
            
            # Calculate feature importance for each cluster
            feature_importances = {}
            
            for feature_idx, feature_name in enumerate(self.feature_columns):
                # Calculate cluster centroids for this feature
                feature_centers = [kmeans.cluster_centers_[i][feature_idx] for i in range(n_clusters)]
                
                # Calculate feature variance across clusters
                feature_variance = np.var(feature_centers)
                
                feature_importances[feature_name] = feature_variance
            
            # Normalize feature importances
            if feature_importances:
                total_variance = sum(feature_importances.values())
                if total_variance > 0:
                    feature_importances = {k: v / total_variance for k, v in feature_importances.items()}
            
            # Apply dimensionality reduction for visualization
            if len(X_transformed) > 10:
                # Use t-SNE for visualization
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_transformed) - 1))
                X_tsne = tsne.fit_transform(X_transformed)
                
                # Create visualization data
                viz_data = {
                    'x': X_tsne[:, 0].tolist(),
                    'y': X_tsne[:, 1].tolist(),
                    'cluster': clusters.tolist(),
                    'success': (trade_data['pnl'] > 0).astype(int).tolist() if 'pnl' in trade_data.columns else None,
                    'pnl': trade_data['pnl'].tolist() if 'pnl' in trade_data.columns else None,
                    'symbol': trade_data['symbol'].tolist() if 'symbol' in trade_data.columns else None
                }
            else:
                viz_data = None
            
            # Prepare results
            results = {
                'clusters': clusters.tolist(),
                'cluster_stats': cluster_stats,
                'feature_importances': feature_importances,
                'visualization_data': viz_data,
                'n_clusters': n_clusters,
                'n_trades': len(trade_data)
            }
            
            # Visualize clusters
            self._visualize_clusters(trade_data, clusters, viz_data, cluster_stats)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error clustering trades: {str(e)}")
            return None
    
    def _visualize_clusters(self, trade_data, clusters, viz_data, cluster_stats):
        """
        Visualize trade clusters.
        
        Args:
            trade_data (pd.DataFrame): Trade data
            clusters (numpy.ndarray): Cluster assignments
            viz_data (dict): Visualization data
            cluster_stats (dict): Cluster statistics
            
        Returns:
            tuple: Figure and axes
        """
        try:
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 14))
            
            # Plot cluster visualization if available
            if viz_data is not None:
                scatter = axs[0, 0].scatter(
                    viz_data['x'],
                    viz_data['y'],
                    c=clusters,
                    cmap='viridis',
                    alpha=0.8,
                    s=50
                )
                
                axs[0, 0].set_title('Trade Clusters Visualization', fontsize=14)
                axs[0, 0].set_xlabel('t-SNE Component 1')
                axs[0, 0].set_ylabel('t-SNE Component 2')
                
                # Add legend
                legend1 = axs[0, 0].legend(*scatter.legend_elements(),
                                        title="Clusters")
                axs[0, 0].add_artist(legend1)
            else:
                axs[0, 0].text(0.5, 0.5, "Insufficient data for visualization", 
                             ha='center', va='center', transform=axs[0, 0].transAxes)
            
            # Plot cluster success rates
            if all('success_rate' in stats and stats['success_rate'] is not None for stats in cluster_stats.values()):
                cluster_ids = list(cluster_stats.keys())
                success_rates = [cluster_stats[c]['success_rate'] for c in cluster_ids]
                
                bars = axs[0, 1].bar(cluster_ids, success_rates)
                
                # Color bars by success rate
                for i, bar in enumerate(bars):
                    bar.set_color(plt.cm.RdYlGn(success_rates[i]))
                
                axs[0, 1].set_title('Success Rate by Cluster', fontsize=14)
                axs[0, 1].set_xlabel('Cluster')
                axs[0, 1].set_ylabel('Success Rate')
                axs[0, 1].set_ylim(0, 1)
                
                # Add value labels
                for i, v in enumerate(success_rates):
                    axs[0, 1].text(i, v + 0.02, f"{v:.2f}", ha='center')
            else:
                axs[0, 1].text(0.5, 0.5, "No trade data available", 
                             ha='center', va='center', transform=axs[0, 1].transAxes)
            
            # Plot exit timing by feature
            if 'exit_timing_by_feature' in results and 'option_type' in results['exit_timing_by_feature']:
                feature_data = results['exit_timing_by_feature']['option_type']
                
                # Check if we have data for both calls and puts
                if 'call' in feature_data and 'put' in feature_data:
                    # Get common days
                    call_days = set(day for day in feature_data['call'].keys() if isinstance(day, int))
                    put_days = set(day for day in feature_data['put'].keys() if isinstance(day, int))
                    common_days = sorted(call_days.intersection(put_days))
                    
                    if common_days:
                        call_returns = [feature_data['call'][day] for day in common_days]
                        put_returns = [feature_data['put'][day] for day in common_days]
                        
                        axs[1, 0].plot(common_days, call_returns, 'g-', label='Calls', linewidth=2)
                        axs[1, 0].plot(common_days, put_returns, 'r-', label='Puts', linewidth=2)
                        
                        # Add optimal points
                        if 'optimal_day' in feature_data['call']:
                            optimal_day = feature_data['call']['optimal_day']
                            if optimal_day in feature_data['call']:
                                optimal_return = feature_data['call'][optimal_day]
                                axs[1, 0].scatter([optimal_day], [optimal_return], c='darkgreen', s=100)
                        
                        if 'optimal_day' in feature_data['put']:
                            optimal_day = feature_data['put']['optimal_day']
                            if optimal_day in feature_data['put']:
                                optimal_return = feature_data['put'][optimal_day]
                                axs[1, 0].scatter([optimal_day], [optimal_return], c='darkred', s=100)
                        
                        axs[1, 0].set_title('Exit Timing by Option Type', fontsize=14)
                        axs[1, 0].set_xlabel('Holding Days')
                        axs[1, 0].set_ylabel('Average Return')
                        axs[1, 0].legend()
                        axs[1, 0].grid(True)
                    else:
                        axs[1, 0].text(0.5, 0.5, "Insufficient data for option type analysis", 
                                     ha='center', va='center', transform=axs[1, 0].transAxes)
                else:
                    axs[1, 0].text(0.5, 0.5, "Missing data for calls or puts", 
                                 ha='center', va='center', transform=axs[1, 0].transAxes)
            else:
                axs[1, 0].text(0.5, 0.5, "Option type data not available", 
                             ha='center', va='center', transform=axs[1, 0].transAxes)
            
            # Plot actual vs. optimal exit comparison
            if trade_returns:
                actual_days = [trade['actual_holding_days'] for trade in trade_returns 
                             if 'actual_holding_days' in trade and 'optimal_exit_day' in trade]
                optimal_days = [trade['optimal_exit_day'] for trade in trade_returns 
                              if 'actual_holding_days' in trade and 'optimal_exit_day' in trade]
                
                if actual_days and optimal_days:
                    # Create comparison data
                    comparison_data = pd.DataFrame({
                        'Actual': actual_days,
                        'Optimal': optimal_days
                    })
                    
                    # Calculate average actual and optimal
                    avg_actual = np.mean(actual_days)
                    avg_optimal = np.mean(optimal_days)
                    
                    # Create scatter plot
                    axs[1, 1].scatter(actual_days, optimal_days, alpha=0.5)
                    axs[1, 1].plot([0, max(actual_days)], [0, max(actual_days)], 'k--')  # Diagonal line
                    
                    # Add means
                    axs[1, 1].axvline(x=avg_actual, color='blue', linestyle='--', 
                                     label=f'Avg Actual: {avg_actual:.1f}d')
                    axs[1, 1].axhline(y=avg_optimal, color='red', linestyle='--',
                                     label=f'Avg Optimal: {avg_optimal:.1f}d')
                    
                    axs[1, 1].set_title('Actual vs. Optimal Holding Period', fontsize=14)
                    axs[1, 1].set_xlabel('Actual Holding Days')
                    axs[1, 1].set_ylabel('Optimal Holding Days')
                    axs[1, 1].legend()
                    axs[1, 1].grid(True)
                else:
                    axs[1, 1].text(0.5, 0.5, "Insufficient data for exit comparison", 
                                 ha='center', va='center', transform=axs[1, 1].transAxes)
            else:
                axs[1, 1].text(0.5, 0.5, "No trade data available", 
                             ha='center', va='center', transform=axs[1, 1].transAxes)
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.models_dir}/exit_timing_analysis_{timestamp}.png"
            plt.savefig(plot_file)
            
            self.logger.info(f"Exit timing analysis visualization saved to {plot_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing exit timing: {str(e)}")
            return None, None
0.5, 0.5, "Success rate data not available", 
                             ha='center', va='center', transform=axs[0, 1].transAxes)
            
            # Plot average PnL by cluster
            if all('avg_pnl' in stats and stats['avg_pnl'] is not None for stats in cluster_stats.values()):
                cluster_ids = list(cluster_stats.keys())
                avg_pnls = [cluster_stats[c]['avg_pnl'] for c in cluster_ids]
                
                bars = axs[1, 0].bar(cluster_ids, avg_pnls)
                
                # Color bars by PnL
                for i, bar in enumerate(bars):
                    bar.set_color('green' if avg_pnls[i] > 0 else 'red')
                
                axs[1, 0].set_title('Average PnL by Cluster', fontsize=14)
                axs[1, 0].set_xlabel('Cluster')
                axs[1, 0].set_ylabel('Average PnL')
                
                # Add value labels
                for i, v in enumerate(avg_pnls):
                    axs[1, 0].text(i, v + (0.02 * max(avg_pnls) if v > 0 else -0.02 * min(avg_pnls)), 
                                 f"${v:.2f}", ha='center')
            else:
                axs[1, 0].text(0.5, 0.5, "PnL data not available", 
                             ha='center', va='center', transform=axs[1, 0].transAxes)
            
            # Plot cluster sizes
            cluster_ids = list(cluster_stats.keys())
            counts = [cluster_stats[c]['count'] for c in cluster_ids]
            
            axs[1, 1].pie(counts, labels=cluster_ids, autopct='%1.1f%%', 
                        startangle=90, colors=plt.cm.viridis(np.linspace(0, 1, len(cluster_ids))))
            
            axs[1, 1].set_title('Cluster Sizes', fontsize=14)
            axs[1, 1].axis('equal')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.models_dir}/trade_clusters_{timestamp}.png"
            plt.savefig(plot_file)
            
            self.logger.info(f"Cluster visualization saved to {plot_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing clusters: {str(e)}")
            return None, None
    
    def optimize_exit_timing(self, trade_data, price_data, max_holding_days=10):
        """
        Analyze and optimize exit timing based on historical trades.
        
        Args:
            trade_data (pd.DataFrame): Historical trade data
            price_data (pd.DataFrame): Historical price data
            max_holding_days (int): Maximum holding period to analyze
            
        Returns:
            dict: Exit timing analysis results
        """
        self.logger.info("Analyzing optimal exit timing")
        
        try:
            # Ensure we have required columns
            required_trade_cols = ['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl']
            if not all(col in trade_data.columns for col in required_trade_cols):
                missing = [col for col in required_trade_cols if col not in trade_data.columns]
                self.logger.error(f"Missing required trade columns: {missing}")
                return None
                
            required_price_cols = ['symbol', 'date', 'close']
            if not all(col in price_data.columns for col in required_price_cols):
                missing = [col for col in required_price_cols if col not in price_data.columns]
                self.logger.error(f"Missing required price columns: {missing}")
                return None
            
            # Ensure date columns are datetime
            for df, date_cols in [(trade_data, ['entry_date', 'exit_date']), (price_data, ['date'])]:
                for col in date_cols:
                    if not pd.api.types.is_datetime64_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col])
            
            # Results structure
            results = {
                'optimal_holding_days': {},
                'exit_timing_by_feature': {},
                'overall_optimal_days': None,
                'exit_timing_model': None
            }
            
            # Analyze each trade's potential returns over different holding periods
            trade_returns = []
            
            for _, trade in trade_data.iterrows():
                symbol = trade['symbol']
                entry_date = trade['entry_date']
                actual_exit_date = trade['exit_date']
                entry_price = trade['entry_price']
                actual_pnl = trade['pnl']
                
                # Get price data for this symbol during the trade period
                symbol_prices = price_data[
                    (price_data['symbol'] == symbol) & 
                    (price_data['date'] >= entry_date) & 
                    (price_data['date'] <= entry_date + timedelta(days=max_holding_days))
                ].sort_values('date')
                
                if len(symbol_prices) <= 1:
                    continue
                
                # Calculate returns for each potential holding period
                trade_results = {
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'actual_exit_date': actual_exit_date,
                    'actual_holding_days': (actual_exit_date - entry_date).days,
                    'actual_pnl': actual_pnl,
                    'returns_by_day': {}
                }
                
                # Add trade features that might affect optimal exit timing
                for feature in ['option_type', 'moneyness', 'days_to_expiration_at_entry']:
                    if feature in trade:
                        trade_results[feature] = trade[feature]
                
                # Calculate returns for each day
                for i, (_, price_row) in enumerate(symbol_prices.iterrows()):
                    day = (price_row['date'] - entry_date).days
                    if day == 0:  # Skip entry day
                        continue
                        
                    # Calculate return
                    day_return = (price_row['close'] - entry_price) / entry_price
                    
                    trade_results['returns_by_day'][day] = day_return
                
                # Find optimal exit day
                if trade_results['returns_by_day']:
                    optimal_day = max(trade_results['returns_by_day'].items(), key=lambda x: x[1])[0]
                    optimal_return = trade_results['returns_by_day'][optimal_day]
                    
                    trade_results['optimal_exit_day'] = optimal_day
                    trade_results['optimal_return'] = optimal_return
                    trade_results['return_improvement'] = optimal_return - (actual_pnl / entry_price)
                    
                    trade_returns.append(trade_results)
            
            if not trade_returns:
                self.logger.warning("No valid trades for exit timing analysis")
                return None
            
            # Calculate overall optimal holding period
            all_returns_by_day = {}
            
            for trade in trade_returns:
                for day, ret in trade['returns_by_day'].items():
                    if day not in all_returns_by_day:
                        all_returns_by_day[day] = []
                    all_returns_by_day[day].append(ret)
            
            avg_returns_by_day = {day: np.mean(returns) for day, returns in all_returns_by_day.items()}
            
            # Find overall optimal day
            if avg_returns_by_day:
                overall_optimal_day = max(avg_returns_by_day.items(), key=lambda x: x[1])[0]
                results['overall_optimal_days'] = overall_optimal_day
            
            # Analyze by features
            feature_analysis = {}
            
            for feature in ['option_type', 'moneyness']:
                if any(feature in trade for trade in trade_returns):
                    feature_returns = {}
                    
                    for trade in trade_returns:
                        if feature in trade and trade[feature] is not None:
                            feature_value = trade[feature]
                            
                            # Discretize continuous features
                            if feature == 'moneyness':
                                if feature_value < -0.05:
                                    feature_value = 'OTM'
                                elif feature_value > 0.05:
                                    feature_value = 'ITM'
                                else:
                                    feature_value = 'ATM'
                            
                            if feature_value not in feature_returns:
                                feature_returns[feature_value] = {}
                            
                            for day, ret in trade['returns_by_day'].items():
                                if day not in feature_returns[feature_value]:
                                    feature_returns[feature_value][day] = []
                                feature_returns[feature_value][day].append(ret)
                    
                    # Calculate average returns by day for each feature value
                    feature_avg_returns = {}
                    
                    for value, returns_by_day in feature_returns.items():
                        feature_avg_returns[value] = {
                            day: np.mean(returns) for day, returns in returns_by_day.items()
                        }
                        
                        # Find optimal day for this feature value
                        if feature_avg_returns[value]:
                            optimal_day = max(feature_avg_returns[value].items(), key=lambda x: x[1])[0]
                            feature_avg_returns[value]['optimal_day'] = optimal_day
                    
                    feature_analysis[feature] = feature_avg_returns
            
            results['exit_timing_by_feature'] = feature_analysis
            
            # Visualize exit timing analysis
            self._visualize_exit_timing(results, trade_returns)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing exit timing: {str(e)}")
            return None
    
    def _visualize_exit_timing(self, results, trade_returns):
        """
        Visualize exit timing analysis.
        
        Args:
            results (dict): Exit timing analysis results
            trade_returns (list): List of trade return data
            
        Returns:
            tuple: Figure and axes
        """
        try:
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 14))
            
            # Plot average returns by holding period
            if 'overall_optimal_days' in results and results['overall_optimal_days'] is not None:
                all_returns_by_day = {}
                
                for trade in trade_returns:
                    for day, ret in trade['returns_by_day'].items():
                        if day not in all_returns_by_day:
                            all_returns_by_day[day] = []
                        all_returns_by_day[day].append(ret)
                
                avg_returns = {day: np.mean(returns) for day, returns in all_returns_by_day.items()}
                
                # Sort by day
                days = sorted(avg_returns.keys())
                returns = [avg_returns[day] for day in days]
                
                axs[0, 0].plot(days, returns, 'o-', linewidth=2)
                axs[0, 0].set_title('Average Returns by Holding Period', fontsize=14)
                axs[0, 0].set_xlabel('Holding Days')
                axs[0, 0].set_ylabel('Average Return')
                axs[0, 0].grid(True)
                
                # Highlight optimal day
                optimal_day = results['overall_optimal_days']
                optimal_return = avg_returns[optimal_day]
                
                axs[0, 0].scatter([optimal_day], [optimal_return], c='red', s=100)
                axs[0, 0].annotate(f"Optimal: {optimal_day} days",
                                 (optimal_day, optimal_return),
                                 xytext=(10, 10),
                                 textcoords='offset points',
                                 fontsize=12,
                                 arrowprops=dict(arrowstyle='->', lw=1.5))
            else:
                axs[0, 0].text(0.5, 0.5, "Insufficient data for analysis", 
                             ha='center', va='center', transform=axs[0, 0].transAxes)
            
            # Plot return distribution by holding period
            if trade_returns:
                # Collect all returns by day
                all_day_returns = {}
                
                for trade in trade_returns:
                    for day, ret in trade['returns_by_day'].items():
                        if day not in all_day_returns:
                            all_day_returns[day] = []
                        all_day_returns[day].append(ret)
                
                # Filter to days with enough data
                all_day_returns = {day: returns for day, returns in all_day_returns.items() if len(returns) >= 5}
                
                if all_day_returns:
                    # Create boxplot data
                    box_data = [all_day_returns[day] for day in sorted(all_day_returns.keys())]
                    
                    axs[0, 1].boxplot(box_data, labels=sorted(all_day_returns.keys()))
                    axs[0, 1].set_title('Return Distribution by Holding Period', fontsize=14)
                    axs[0, 1].set_xlabel('Holding Days')
                    axs[0, 1].set_ylabel('Return')
                    axs[0, 1].grid(True)
                else:
                    axs[0, 1].text(0.5, 0.5, "Insufficient data for distribution analysis", 
                                 ha='center', va='center', transform=axs[0, 1].transAxes)
            else:
                axs[0, 1].text(                    'n_estimators': [50,"""
