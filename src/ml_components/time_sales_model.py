"""
Time Sales Model Module

This module implements machine learning models for analyzing time & sales data
to predict short-term price movements and detect unusual trading patterns.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class TimeSalesModel:
    """
    Machine learning model for analyzing time & sales data to predict price movements.
    
    Features:
    - Time-series pattern detection
    - Volume spike prediction
    - Abnormal trade flow detection
    - Block trade identification
    """
    
    def __init__(self, config=None):
        """
        Initialize the TimeSalesModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = "models/time_sales"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.model_type = self.config.get('time_sales_model_type', 'xgboost')
        self.lookback_window = self.config.get('time_sales_lookback', 20)
        self.prediction_horizon = self.config.get('time_sales_prediction_horizon', 5)
        self.threshold = self.config.get('time_sales_threshold', 0.6)
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        
        self.logger.info(f"TimeSalesModel initialized with {self.model_type} model")
    
    def preprocess_data(self, time_sales_data):
        """
        Preprocess time & sales data for model training.
        
        Args:
            time_sales_data (pd.DataFrame): Raw time & sales data
            
        Returns:
            tuple: X (features) and y (targets) for model training
        """
        self.logger.info("Preprocessing time & sales data")
        
        # Check if we have the required columns
        required_columns = ['time', 'price', 'size', 'exchange', 'condition']
        
        if not all(col in time_sales_data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in time_sales_data.columns]
            self.logger.error(f"Missing required columns: {missing}")
            return None, None
        
        # Ensure time is in datetime format
        if not pd.api.types.is_datetime64_dtype(time_sales_data['time']):
            time_sales_data['time'] = pd.to_datetime(time_sales_data['time'])
        
        # Sort by time
        time_sales_data = time_sales_data.sort_values('time')
        
        # Extract basic features
        df = time_sales_data.copy()
        
        # Add price movement features
        df['price_diff'] = df['price'].diff()
        df['price_pct_change'] = df['price'].pct_change()
        
        # Add volume features
        df['log_size'] = np.log1p(df['size'])
        df['size_diff'] = df['size'].diff()
        df['size_pct_change'] = df['size'].pct_change()
        
        # Add time interval features
        df['time_diff'] = df['time'].diff().dt.total_seconds()
        
        # Add rolling stats
        for window in [5, 10, 20]:
            # Price stats
            df[f'price_mean_{window}'] = df['price'].rolling(window).mean()
            df[f'price_std_{window}'] = df['price'].rolling(window).std()
            df[f'price_z_{window}'] = (df['price'] - df[f'price_mean_{window}']) / df[f'price_std_{window}'].replace(0, 1)
            
            # Volume stats
            df[f'size_mean_{window}'] = df['size'].rolling(window).mean()
            df[f'size_std_{window}'] = df['size'].rolling(window).std()
            df[f'size_z_{window}'] = (df['size'] - df[f'size_mean_{window}']) / df[f'size_std_{window}'].replace(0, 1)
            
            # Time interval stats
            df[f'time_diff_mean_{window}'] = df['time_diff'].rolling(window).mean()
        
        # Create exchange and condition categorical features
        df['exchange_code'] = df['exchange'].astype('category').cat.codes
        df['condition_code'] = df['condition'].astype('category').cat.codes
        
        # Add trade direction feature
        df['trade_direction'] = np.sign(df['price_diff']).fillna(0)
        
        # Add future price movement (target)
        future_price_changes = []
        for i in range(1, self.prediction_horizon + 1):
            future_price = df['price'].shift(-i)
            price_change = (future_price - df['price']) / df['price']
            future_price_changes.append(price_change)
        
        # Average future price change
        df['future_price_change'] = pd.concat(future_price_changes, axis=1).mean(axis=1)
        
        # Create target labels (1: price goes up, 0: price goes down or stays)
        threshold = 0.001  # 0.1% threshold
        df['target'] = (df['future_price_change'] > threshold).astype(int)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Select features
        features = [
            'price_diff', 'price_pct_change', 'log_size', 'size_diff', 'size_pct_change',
            'time_diff', 'exchange_code', 'condition_code', 'trade_direction',
            'price_mean_5', 'price_std_5', 'price_z_5', 'size_mean_5', 'size_std_5', 'size_z_5',
            'price_mean_10', 'price_std_10', 'price_z_10', 'size_mean_10', 'size_std_10', 'size_z_10',
            'price_mean_20', 'price_std_20', 'price_z_20', 'size_mean_20', 'size_std_20', 'size_z_20',
        ]
        
        X = df[features].values
        y = df['target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.logger.info(f"Preprocessed data: {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def prepare_lstm_sequences(self, X, y, lookback=None):
        """
        Prepare sequences for LSTM model.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector
            lookback (int, optional): Lookback window size
            
        Returns:
            tuple: X_sequences, y_sequences
        """
        if lookback is None:
            lookback = self.lookback_window
        
        X_sequences, y_sequences = [], []
        
        for i in range(len(X) - lookback):
            X_sequences.append(X[i:i+lookback])
            y_sequences.append(y[i+lookback])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self, input_shape=None, model_type=None):
        """
        Build the machine learning model.
        
        Args:
            input_shape (tuple, optional): Input shape for LSTM models
            model_type (str, optional): Type of model to build
            
        Returns:
            object: Trained model
        """
        if model_type is None:
            model_type = self.model_type
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            
        elif model_type == 'lstm':
            if input_shape is None:
                raise ValueError("Input shape must be provided for LSTM model")
                
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.2),
                LSTM(32),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def train(self, time_sales_data, model_name=None, test_size=0.2, epochs=50, batch_size=32):
        """
        Train the time & sales model.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            model_name (str, optional): Name for the saved model
            test_size (float): Proportion of data to use for testing
            epochs (int): Number of training epochs (for neural networks)
            batch_size (int): Batch size for training (for neural networks)
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training {self.model_type} model on time & sales data")
        
        # Preprocess data
        X, y = self.preprocess_data(time_sales_data)
        
        if X is None or y is None:
            self.logger.error("Preprocessing failed, cannot train model")
            return None
        
        # Split data into train and test sets
        if self.model_type == 'lstm':
            # Prepare sequences for LSTM
            X_seq, y_seq = self.prepare_lstm_sequences(X, y)
            
            # Split by time
            train_samples = int(len(X_seq) * (1 - test_size))
            X_train, X_test = X_seq[:train_samples], X_seq[train_samples:]
            y_train, y_test = y_seq[:train_samples], y_seq[train_samples:]
            
            # Build LSTM model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape=input_shape, model_type='lstm')
            
            # Set up callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
            
            if model_name:
                model_path = os.path.join(self.models_dir, f"{model_name}.h5")
                checkpoint = ModelCheckpoint(model_path, save_best_only=True)
                callbacks = [early_stopping, reduce_lr, checkpoint]
            else:
                callbacks = [early_stopping, reduce_lr]
            
            # Train LSTM model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            metrics = {}
            eval_results = self.model.evaluate(X_test, y_test)
            metrics['loss'] = eval_results[0]
            metrics['accuracy'] = eval_results[1]
            
            # Make predictions
            y_prob = self.model.predict(X_test)
            y_pred = (y_prob > 0.5).astype(int).reshape(-1)
            
        else:
            # Train-test split for non-sequential models
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Build and train model
            self.model = self.build_model(model_type=self.model_type)
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_prob = None
            
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)[:, 1]
            
            # Evaluation metrics
            metrics = {}
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Detailed classification metrics
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        metrics['precision'] = p
        metrics['recall'] = r
        metrics['f1_score'] = f1
        
        # Save feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importances'] = self.model.feature_importances_.tolist()
        
        # Save model
        if model_name:
            if self.model_type != 'lstm':  # LSTM is saved via callback
                model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'scaler': self.scaler,
                        'config': {
                            'model_type': self.model_type,
                            'lookback_window': self.lookback_window,
                            'prediction_horizon': self.prediction_horizon,
                            'threshold': self.threshold
                        }
                    }, f)
                
                self.logger.info(f"Model saved to {model_path}")
        
        self.logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
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
            if model_path.endswith('.h5'):
                # Load LSTM model
                self.model = load_model(model_path)
                self.model_type = 'lstm'
                
                # Try to load config from accompanying JSON file
                config_path = model_path.replace('.h5', '_config.json')
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        self.lookback_window = config.get('lookback_window', self.lookback_window)
                        self.prediction_horizon = config.get('prediction_horizon', self.prediction_horizon)
                        self.threshold = config.get('threshold', self.threshold)
                
                self.logger.info(f"Loaded LSTM model from {model_path}")
                return True
            else:
                # Load pickle model
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                
                # Load configuration
                if 'config' in saved_data:
                    config = saved_data['config']
                    self.model_type = config.get('model_type', self.model_type)
                    self.lookback_window = config.get('lookback_window', self.lookback_window)
                    self.prediction_horizon = config.get('prediction_horizon', self.prediction_horizon)
                    self.threshold = config.get('threshold', self.threshold)
                
                self.logger.info(f"Loaded {self.model_type} model from {model_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def predict(self, time_sales_data):
        """
        Make predictions on new time & sales data.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        self.logger.info("Making predictions on time & sales data")
        
        try:
            # Preprocess data
            X, _ = self.preprocess_data(time_sales_data)
            
            if X is None:
                self.logger.error("Preprocessing failed, cannot make predictions")
                return None
            
            # Make prediction
            if self.model_type == 'lstm':
                # Prepare sequences for LSTM
                lookback = self.lookback_window
                
                # Use the last lookback samples for prediction
                if len(X) < lookback:
                    self.logger.warning(f"Not enough data for LSTM prediction, need at least {lookback} samples")
                    return None
                
                X_seq = np.array([X[-lookback:]])
                y_prob = self.model.predict(X_seq)[0][0]
                prediction = 1 if y_prob > self.threshold else 0
                
            else:
                # Use the most recent data point for prediction
                X_latest = X[-1:].reshape(1, -1)
                
                # Get prediction
                prediction = self.model.predict(X_latest)[0]
                
                # Get probability if available
                y_prob = None
                if hasattr(self.model, 'predict_proba'):
                    y_prob = self.model.predict_proba(X_latest)[0, 1]
            
            # Create result dictionary
            result = {
                'prediction': int(prediction),
                'probability': float(y_prob) if y_prob is not None else None,
                'direction': 'up' if prediction == 1 else 'down',
                'confidence': float(abs(0.5 - y_prob) * 2) if y_prob is not None else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Prediction: {result['direction']} with {result.get('confidence', 'unknown')} confidence")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def detect_anomalies(self, time_sales_data, zscore_threshold=3.0):
        """
        Detect anomalies in time & sales data.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            zscore_threshold (float): Z-score threshold for anomaly detection
            
        Returns:
            pd.DataFrame: Anomalies detected in the data
        """
        self.logger.info("Detecting anomalies in time & sales data")
        
        try:
            # Ensure time is in datetime format
            df = time_sales_data.copy()
            if not pd.api.types.is_datetime64_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            # Sort by time
            df = df.sort_values('time')
            
            # Calculate rolling stats
            df['size_mean'] = df['size'].rolling(window=20).mean()
            df['size_std'] = df['size'].rolling(window=20).std()
            df['size_zscore'] = (df['size'] - df['size_mean']) / df['size_std'].replace(0, 1)
            
            df['price_mean'] = df['price'].rolling(window=20).mean()
            df['price_std'] = df['price'].rolling(window=20).std()
            df['price_zscore'] = (df['price'] - df['price_mean']) / df['price_std'].replace(0, 1)
            
            # Add time interval
            df['time_diff'] = df['time'].diff().dt.total_seconds()
            df['time_diff_mean'] = df['time_diff'].rolling(window=20).mean()
            df['time_diff_std'] = df['time_diff'].rolling(window=20).std()
            df['time_diff_zscore'] = (df['time_diff'] - df['time_diff_mean']) / df['time_diff_std'].replace(0, 1)
            
            # Detect anomalies
            volume_anomalies = df[abs(df['size_zscore']) > zscore_threshold].copy()
            price_anomalies = df[abs(df['price_zscore']) > zscore_threshold].copy()
            time_anomalies = df[abs(df['time_diff_zscore']) > zscore_threshold].copy()
            
            # Combine anomalies
            all_anomalies = pd.concat([volume_anomalies, price_anomalies, time_anomalies])
            all_anomalies = all_anomalies.drop_duplicates().sort_values('time')
            
            # Add anomaly type
            all_anomalies['anomaly_type'] = ''
            all_anomalies.loc[abs(all_anomalies['size_zscore']) > zscore_threshold, 'anomaly_type'] += 'volume_'
            all_anomalies.loc[abs(all_anomalies['price_zscore']) > zscore_threshold, 'anomaly_type'] += 'price_'
            all_anomalies.loc[abs(all_anomalies['time_diff_zscore']) > zscore_threshold, 'anomaly_type'] += 'time_'
            all_anomalies['anomaly_type'] = all_anomalies['anomaly_type'].str.rstrip('_')
            
            self.logger.info(f"Detected {len(all_anomalies)} anomalies in time & sales data")
            
            return all_anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return pd.DataFrame()
    
    def visualize_predictions(self, time_sales_data, window=100):
        """
        Visualize time & sales data with predictions.
        
        Args:
            time_sales_data (pd.DataFrame): Time & sales data
            window (int): Number of recent data points to visualize
            
        Returns:
            tuple: Figure and axes
        """
        if self.model is None:
            self.logger.error("Model not trained or loaded")
            return None, None
        
        try:
            # Preprocess data
            X, _ = self.preprocess_data(time_sales_data)
            
            if X is None:
                self.logger.error("Preprocessing failed, cannot visualize predictions")
                return None, None
            
            # Make predictions on all data
            if self.model_type == 'lstm':
                # Prepare sequences for LSTM
                lookback = self.lookback_window
                
                predictions = []
                probabilities = []
                
                for i in range(lookback, len(X)):
                    X_seq = np.array([X[i-lookback:i]])
                    prob = self.model.predict(X_seq)[0][0]
                    pred = 1 if prob > self.threshold else 0
                    
                    predictions.append(pred)
                    probabilities.append(prob)
                
                # Align with original data
                pred_indices = range(lookback, len(X))
                
            else:
                # Use all data points
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)[:, 1]
                    predictions = (probabilities > self.threshold).astype(int)
                else:
                    predictions = self.model.predict(X)
                    probabilities = None
                
                pred_indices = range(len(X))
            
            # Get recent window of data
            if window and window < len(time_sales_data):
                recent_data = time_sales_data[-window:].copy()
            else:
                recent_data = time_sales_data.copy()
            
            # Create visualization
            fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            
            # Plot price
            axs[0].plot(recent_data['time'], recent_data['price'], label='Price')
            axs[0].set_title('Time & Sales Data with Predictions')
            axs[0].set_ylabel('Price')
            axs[0].grid(True)
            
            # Plot volume
            axs[1].bar(recent_data['time'], recent_data['size'], label='Volume')
            axs[1].set_ylabel('Volume')
            axs[1].grid(True)
            
            # Plot predictions
            if len(pred_indices) > 0:
                pred_times = [time_sales_data['time'].iloc[i] for i in pred_indices if i < len(time_sales_data)]
                pred_prices = [time_sales_data['price'].iloc[i] for i in pred_indices if i < len(time_sales_data)]
                
                # Filter to recent window
                if window and window < len(pred_times):
                    start_time = recent_data['time'].iloc[0]
                    mask = [t >= start_time for t in pred_times]
                    pred_times = [pred_times[i] for i in range(len(pred_times)) if mask[i]]
                    pred_prices = [pred_prices[i] for i in range(len(pred_prices)) if mask[i]]
                    predictions = [predictions[i] for i in range(len(predictions)) if i < len(mask) and mask[i]]
                    
                    if probabilities is not None:
                        probabilities = [probabilities[i] for i in range(len(probabilities)) if i < len(mask) and mask[i]]
                
                # Plot prediction markers
                for i in range(len(pred_times)):
                    if i < len(predictions):
                        color = 'green' if predictions[i] == 1 else 'red'
                        axs[0].scatter(pred_times[i], pred_prices[i], color=color, s=30)
                
                # Plot prediction probabilities
                if probabilities is not None and len(probabilities) > 0:
                    axs[2].plot(pred_times[:len(probabilities)], probabilities, label='Prediction Probability')
                    axs[2].axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
                    axs[2].set_ylabel('Probability')
                    axs[2].set_ylim(0, 1)
                    axs[2].grid(True)
                    axs[2].legend()
            
            plt.tight_layout()
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing predictions: {str(e)}")
            return None, None
