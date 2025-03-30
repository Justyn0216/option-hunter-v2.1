"""
Order Flow Model Module

This module implements machine learning models for analyzing option order flow data
to detect unusual activity, predict directional movements, and identify potential opportunities.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb

class OrderFlowModel:
    """
    Machine learning model for analyzing option order flow data to predict market movements.
    
    Features:
    - Unusual option activity detection
    - Order flow sentiment analysis
    - Large trade detection and classification
    - Order imbalance prediction
    """
    
    def __init__(self, config=None):
        """
        Initialize the OrderFlowModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = "models/order_flow"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.model_type = self.config.get('order_flow_model_type', 'lightgbm')
        self.prediction_window = self.config.get('order_flow_prediction_window', 3)  # days
        self.threshold = self.config.get('order_flow_threshold', 0.65)
        
        # Initialize model
        self.model = None
        self.preprocessor = None
        
        self.logger.info(f"OrderFlowModel initialized with {self.model_type} model")
    
    def preprocess_data(self, order_flow_data, price_data=None):
        """
        Preprocess order flow data for model training.
        
        Args:
            order_flow_data (pd.DataFrame): Raw order flow data
            price_data (pd.DataFrame, optional): Historical price data for target creation
            
        Returns:
            tuple: X (features) and y (targets) for model training
        """
        self.logger.info("Preprocessing order flow data")
        
        # Check if we have the required columns
        required_columns = ['date', 'symbol', 'option_symbol', 'strike', 'expiration', 
                          'option_type', 'volume', 'open_interest', 'price', 'side']
        
        if not all(col in order_flow_data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in order_flow_data.columns]
            self.logger.error(f"Missing required columns: {missing}")
            return None, None
        
        # Ensure date is in datetime format
        if not pd.api.types.is_datetime64_dtype(order_flow_data['date']):
            order_flow_data['date'] = pd.to_datetime(order_flow_data['date'])
        
        # Convert expiration to datetime format
        if not pd.api.types.is_datetime64_dtype(order_flow_data['expiration']):
            order_flow_data['expiration'] = pd.to_datetime(order_flow_data['expiration'])
        
        # Extract features from order flow data
        df = order_flow_data.copy()
        
        # Calculate days to expiration
        df['days_to_expiration'] = (df['expiration'] - df['date']).dt.days
        
        # Calculate price ratios and moneyness
        if 'underlying_price' in df.columns:
            df['strike_ratio'] = df['strike'] / df['underlying_price']
            
            # Calculate moneyness (how far in/out of the money)
            df['moneyness'] = np.where(
                df['option_type'] == 'call',
                df['underlying_price'] / df['strike'] - 1,
                df['strike'] / df['underlying_price'] - 1
            )
        else:
            # Approximate using option price and delta if available
            if 'delta' in df.columns:
                # Rough approximation of moneyness using delta
                df['moneyness'] = np.where(
                    df['option_type'] == 'call',
                    df['delta'] * 2 - 1,
                    (1 - df['delta']) * 2 - 1
                )
            else:
                df['moneyness'] = 0
                
            df['strike_ratio'] = 1.0
        
        # Categorical variables encoding
        df['option_type_code'] = (df['option_type'] == 'call').astype(int)  # 1 for call, 0 for put
        df['side_code'] = (df['side'] == 'buy').astype(int)  # 1 for buy, 0 for sell
        
        # Extract day of week and time of day
        df['day_of_week'] = df['date'].dt.dayofweek
        if 'time' in df.columns:
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            df['minute'] = pd.to_datetime(df['time']).dt.minute
            df['time_of_day'] = df['hour'] + df['minute']/60.0
        else:
            df['hour'] = 12  # Default to midday if no time
            df['time_of_day'] = 12
        
        # Process volume and open interest
        df['volume_oi_ratio'] = df['volume'] / df['open_interest'].replace(0, 1)
        df['log_volume'] = np.log1p(df['volume'])
        df['log_open_interest'] = np.log1p(df['open_interest'])
        
        # Calculate premium
        if 'premium' not in df.columns and 'volume' in df.columns and 'price' in df.columns:
            df['premium'] = df['volume'] * df['price'] * 100  # 100 shares per contract
        
        df['log_premium'] = np.log1p(df['premium']) if 'premium' in df.columns else 0
        
        # Aggregate by symbol and date
        agg_data = df.groupby(['symbol', df['date'].dt.date]).agg({
            'volume': 'sum',
            'premium': 'sum',
            'volume_oi_ratio': 'mean',
            'moneyness': 'mean'
        }).reset_index()
        
        # Create call/put ratio features
        call_data = df[df['option_type'] == 'call'].groupby(['symbol', df['date'].dt.date]).agg({
            'volume': 'sum',
            'premium': 'sum'
        }).reset_index()
        call_data.columns = ['symbol', 'date', 'call_volume', 'call_premium']
        
        put_data = df[df['option_type'] == 'put'].groupby(['symbol', df['date'].dt.date]).agg({
            'volume': 'sum',
            'premium': 'sum'
        }).reset_index()
        put_data.columns = ['symbol', 'date', 'put_volume', 'put_premium']
        
        # Merge call and put data
        agg_data = pd.merge(agg_data, call_data, on=['symbol', 'date'], how='left')
        agg_data = pd.merge(agg_data, put_data, on=['symbol', 'date'], how='left')
        
        # Calculate ratios
        agg_data['call_put_volume_ratio'] = agg_data['call_volume'] / agg_data['put_volume'].replace(0, 1)
        agg_data['call_put_premium_ratio'] = agg_data['call_premium'] / agg_data['put_premium'].replace(0, 1)
        
        # Calculate buy/sell imbalance if available
        if 'side' in df.columns:
            buy_data = df[df['side'] == 'buy'].groupby(['symbol', df['date'].dt.date]).agg({
                'volume': 'sum',
                'premium': 'sum'
            }).reset_index()
            buy_data.columns = ['symbol', 'date', 'buy_volume', 'buy_premium']
            
            sell_data = df[df['side'] == 'sell'].groupby(['symbol', df['date'].dt.date]).agg({
                'volume': 'sum',
                'premium': 'sum'
            }).reset_index()
            sell_data.columns = ['symbol', 'date', 'sell_volume', 'sell_premium']
            
            # Merge buy and sell data
            agg_data = pd.merge(agg_data, buy_data, on=['symbol', 'date'], how='left')
            agg_data = pd.merge(agg_data, sell_data, on=['symbol', 'date'], how='left')
            
            # Calculate buy/sell imbalance
            agg_data['buy_sell_volume_ratio'] = agg_data['buy_volume'] / agg_data['sell_volume'].replace(0, 1)
            agg_data['buy_sell_premium_ratio'] = agg_data['buy_premium'] / agg_data['sell_premium'].replace(0, 1)
        
        # Create rolling averages
        agg_data = agg_data.sort_values(['symbol', 'date'])
        for window in [3, 5, 10]:
            for col in ['volume', 'premium', 'call_put_volume_ratio', 'call_put_premium_ratio']:
                if col in agg_data.columns:
                    agg_data[f'{col}_ma{window}'] = agg_data.groupby('symbol')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    agg_data[f'{col}_std{window}'] = agg_data.groupby('symbol')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    # Z-score
                    agg_data[f'{col}_z{window}'] = (agg_data[col] - agg_data[f'{col}_ma{window}']) / agg_data[f'{col}_std{window}'].replace(0, 1)
            
            if 'buy_sell_volume_ratio' in agg_data.columns:
                agg_data[f'buy_sell_volume_ratio_ma{window}'] = agg_data.groupby('symbol')['buy_sell_volume_ratio'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        # Create the target variable if price data is provided
        if price_data is not None:
            # Ensure price_data has date as datetime
            if not pd.api.types.is_datetime64_dtype(price_data['date']):
                price_data['date'] = pd.to_datetime(price_data['date'])
            
            # Create future price change for each symbol
            price_changes = []
            
            for symbol in agg_data['symbol'].unique():
                symbol_prices = price_data[price_data['symbol'] == symbol].copy()
                if len(symbol_prices) < 2:
                    continue
                    
                symbol_prices = symbol_prices.sort_values('date')
                symbol_prices['future_price'] = symbol_prices['close'].shift(-self.prediction_window)
                symbol_prices['price_change'] = (symbol_prices['future_price'] - symbol_prices['close']) / symbol_prices['close']
                
                # Map to aggregated data dates
                for date in agg_data[agg_data['symbol'] == symbol]['date']:
                    date_dt = pd.to_datetime(date)
                    price_row = symbol_prices[symbol_prices['date'] <= date_dt].iloc[-1] if len(symbol_prices[symbol_prices['date'] <= date_dt]) > 0 else None
                    
                    if price_row is not None and not pd.isna(price_row['price_change']):
                        price_changes.append({
                            'symbol': symbol,
                            'date': date,
                            'price_change': price_row['price_change']
                        })
            
            if price_changes:
                price_change_df = pd.DataFrame(price_changes)
                
                # Merge with aggregated data
                agg_data = pd.merge(agg_data, price_change_df, on=['symbol', 'date'], how='left')
                
                # Create binary target (1 for positive return, 0 for negative)
                threshold = 0.02  # 2% price change threshold
                agg_data['target'] = (agg_data['price_change'] > threshold).astype(int)
                
                # Drop rows with missing targets
                agg_data = agg_data.dropna(subset=['target'])
            else:
                self.logger.warning("Could not create target variable, no matching price data")
                return None, None
        
        # Fill NA values
        agg_data = agg_data.fillna(0)
        
        # Select features
        feature_columns = [col for col in agg_data.columns if col not in [
            'symbol', 'date', 'price_change', 'target'
        ]]
        
        X = agg_data[feature_columns]
        y = agg_data['target'] if 'target' in agg_data.columns else None
        
        self.logger.info(f"Preprocessed data: {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store metadata for later
        self.feature_columns = feature_columns
        
        return X, y
    
    def build_preprocessor(self, X):
        """
        Build a preprocessor for feature transformation.
        
        Args:
            X (pd.DataFrame): Feature DataFrame
            
        Returns:
            ColumnTransformer: Scikit-learn preprocessor
        """
        # Separate numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def build_model(self, model_type=None):
        """
        Build the machine learning model.
        
        Args:
            model_type (str, optional): Type of model to build
            
        Returns:
            object: Model instance
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
            
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def train(self, order_flow_data, price_data=None, model_name=None, test_size=0.2):
        """
        Train the order flow model.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            price_data (pd.DataFrame, optional): Historical price data for target creation
            model_name (str, optional): Name for the saved model
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training {self.model_type} model on order flow data")
        
        # Preprocess data
        X, y = self.preprocess_data(order_flow_data, price_data)
        
        if X is None or y is None:
            self.logger.error("Preprocessing failed, cannot train model")
            return None
        
        # Build preprocessor
        self.preprocessor = self.build_preprocessor(X)
        
        # Chronological split for time series data
        if 'date' in order_flow_data.columns:
            order_flow_data['date'] = pd.to_datetime(order_flow_data['date'])
            dates = pd.to_datetime(order_flow_data['date'].unique()).sort_values()
            
            train_dates = dates[:int(len(dates) * (1 - test_size))]
            test_dates = dates[int(len(dates) * (1 - test_size)):]
            
            train_indices = X.index[order_flow_data['date'].isin(train_dates)]
            test_indices = X.index[order_flow_data['date'].isin(test_dates)]
            
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        else:
            # Regular train-test split if dates are not available
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Preprocess features
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Build and train model
        self.model = self.build_model(model_type=self.model_type)
        self.model.fit(X_train_transformed, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_transformed)
        
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test_transformed)[:, 1]
        
        # Evaluation metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Detailed classification metrics
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        metrics['precision'] = p
        metrics['recall'] = r
        metrics['f1_score'] = f1
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        # Save feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            metrics['feature_importances'] = feature_importance.to_dict()
        
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
                        'prediction_window': self.prediction_window,
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
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model = saved_data['model']
            self.preprocessor = saved_data['preprocessor']
            self.feature_columns = saved_data.get('feature_columns', [])
            
            # Load configuration
            if 'config' in saved_data:
                config = saved_data['config']
                self.model_type = config.get('model_type', self.model_type)
                self.prediction_window = config.get('prediction_window', self.prediction_window)
                self.threshold = config.get('threshold', self.threshold)
            
            self.logger.info(f"Loaded {self.model_type} model from {model_path}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def predict(self, order_flow_data):
        """
        Make predictions on new order flow data.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            
        Returns:
            dict: Prediction results
        """
        if self.model is None or self.preprocessor is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        self.logger.info("Making predictions on order flow data")
        
        try:
            # Preprocess data
            X, _ = self.preprocess_data(order_flow_data)
            
            if X is None:
                self.logger.error("Preprocessing failed, cannot make predictions")
                return None
            
            # Ensure we have the right features
            missing_features = [col for col in self.feature_columns if col not in X.columns]
            for feature in missing_features:
                X[feature] = 0  # Add missing features with default value
            
            X = X[self.feature_columns]  # Reorder columns to match training data
            
            # Transform features
            X_transformed = self.preprocessor.transform(X)
            
            # Make prediction
            y_pred = self.model.predict(X_transformed)
            
            # Get probability if available
            y_prob = None
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_transformed)[:, 1]
            
            # Create result dictionary
            results = []
            
            for i, (_, row) in enumerate(order_flow_data.iterrows()):
                # Skip if index is out of bounds for predictions
                if i >= len(y_pred):
                    continue
                    
                result = {
                    'symbol': row['symbol'] if 'symbol' in row else 'unknown',
                    'date': row['date'].strftime('%Y-%m-%d') if 'date' in row and hasattr(row['date'], 'strftime') else 'unknown',
                    'prediction': int(y_pred[i]),
                    'probability': float(y_prob[i]) if y_prob is not None else None,
                    'direction': 'bullish' if y_pred[i] == 1 else 'bearish',
                    'confidence': float(abs(0.5 - y_prob[i]) * 2) if y_prob is not None else None,
                    'prediction_window': self.prediction_window
                }
                
                results.append(result)
            
            # Summarize by symbol
            symbol_predictions = {}
            
            for result in results:
                symbol = result['symbol']
                if symbol not in symbol_predictions:
                    symbol_predictions[symbol] = []
                
                symbol_predictions[symbol].append(result)
            
            # Compute aggregate prediction for each symbol
            aggregated_results = {}
            
            for symbol, preds in symbol_predictions.items():
                # Average probabilities
                avg_prob = np.mean([p['probability'] for p in preds if p['probability'] is not None])
                
                # Determine majority prediction
                num_bullish = sum(1 for p in preds if p['direction'] == 'bullish')
                majority_prediction = 1 if num_bullish > len(preds) / 2 else 0
                
                aggregated_results[symbol] = {
                    'symbol': symbol,
                    'prediction': majority_prediction,
                    'probability': avg_prob,
                    'direction': 'bullish' if majority_prediction == 1 else 'bearish',
                    'confidence': abs(0.5 - avg_prob) * 2,
                    'prediction_window': self.prediction_window,
                    'num_samples': len(preds)
                }
            
            self.logger.info(f"Generated predictions for {len(aggregated_results)} symbols")
            
            return {
                'detailed_results': results,
                'aggregated_results': aggregated_results,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def detect_unusual_activity(self, order_flow_data, zscore_threshold=2.5):
        """
        Detect unusual activity in order flow data.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            zscore_threshold (float): Z-score threshold for unusual activity
            
        Returns:
            pd.DataFrame: Unusual activity detected
        """
        self.logger.info("Detecting unusual activity in order flow data")
        
        try:
            # Preprocess data
            df = order_flow_data.copy()
            
            # Ensure date is in datetime format
            if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Calculate volume and premium metrics
            df['premium'] = df['volume'] * df['price'] * 100 if 'price' in df.columns else df['volume']
            
            # Group by symbol, date, and option type
            grouped = df.groupby(['symbol', pd.Grouper(key='date', freq='D'), 'option_type']).agg({
                'volume': 'sum',
                'premium': 'sum',
                'open_interest': 'mean'
            }).reset_index()
            
            # Calculate historical averages and standard deviations
            for symbol in grouped['symbol'].unique():
                symbol_data = grouped[grouped['symbol'] == symbol]
                
                for option_type in ['call', 'put']:
                    type_data = symbol_data[symbol_data['option_type'] == option_type]
                    
                    if len(type_data) > 10:  # Need enough history
                        # Calculate rolling stats
                        for metric in ['volume', 'premium']:
                            rolling_mean = type_data[metric].rolling(window=10).mean()
                            rolling_std = type_data[metric].rolling(window=10).std()
                            
                            # Calculate z-scores
                            z_scores = (type_data[metric] - rolling_mean) / rolling_std.replace(0, 1)
                            
                            # Add to original dataframe
                            z_score_col = f"{option_type}_{metric}_zscore"
                            grouped.loc[type_data.index, z_score_col] = z_scores
            
            # Identify unusual activity
            unusual_activity = grouped[
                (abs(grouped.get('call_volume_zscore', 0)) > zscore_threshold) |
                (abs(grouped.get('put_volume_zscore', 0)) > zscore_threshold) |
                (abs(grouped.get('call_premium_zscore', 0)) > zscore_threshold) |
                (abs(grouped.get('put_premium_zscore', 0)) > zscore_threshold)
            ].copy()
            
            # Add unusual activity type
            unusual_activity['unusual_activity_type'] = ''
            
            # Check each type of unusual activity
            for col, activity_type in [
                ('call_volume_zscore', 'high_call_volume'),
                ('put_volume_zscore', 'high_put_volume'),
                ('call_premium_zscore', 'high_call_premium'),
                ('put_premium_zscore', 'high_put_premium')
            ]:
                if col in unusual_activity.columns:
                    mask = unusual_activity[col] > zscore_threshold
                    unusual_activity.loc[mask, 'unusual_activity_type'] += f"{activity_type},"
            
            # Remove trailing comma and filter empty strings
            unusual_activity['unusual_activity_type'] = unusual_activity['unusual_activity_type'].str.rstrip(',')
            unusual_activity = unusual_activity[unusual_activity['unusual_activity_type'] != '']
            
            # Sort by most unusual (highest absolute z-score)
            for col in ['call_volume_zscore', 'put_volume_zscore', 'call_premium_zscore', 'put_premium_zscore']:
                if col in unusual_activity.columns:
                    unusual_activity[f"abs_{col}"] = abs(unusual_activity[col])
            
            max_zscore_cols = [col for col in unusual_activity.columns if col.startswith('abs_') and col.endswith('_zscore')]
            if max_zscore_cols:
                unusual_activity['max_zscore'] = unusual_activity[max_zscore_cols].max(axis=1)
                unusual_activity = unusual_activity.sort_values('max_zscore', ascending=False)
            
            self.logger.info(f"Detected {len(unusual_activity)} instances of unusual activity")
            
            return unusual_activity
            
        except Exception as e:
            self.logger.error(f"Error detecting unusual activity: {str(e)}")
            return pd.DataFrame()
    
    def visualize_predictions(self, predictions, price_data=None):
        """
        Visualize order flow predictions.
        
        Args:
            predictions (dict): Prediction results from predict method
            price_data (pd.DataFrame, optional): Historical price data
            
        Returns:
            tuple: Figure and axes
        """
        if not predictions:
            self.logger.error("No predictions to visualize")
            return None, None
        
        try:
            agg_results = predictions.get('aggregated_results', {})
            
            if not agg_results:
                self.logger.warning("No aggregated predictions to visualize")
                return None, None
            
            # Create visualization
            fig, axs = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot prediction probabilities by symbol
            symbols = list(agg_results.keys())
            probs = [agg_results[s]['probability'] for s in symbols]
            
            # Sort by probability
            sorted_indices = np.argsort(probs)
            symbols = [symbols[i] for i in sorted_indices]
            probs = [probs[i] for i in sorted_indices]
            
            # Create colormap based on prediction (bullish=green, bearish=red)
            colors = ['green' if agg_results[s]['direction'] == 'bullish' else 'red' for s in symbols]
            
            axs[0].barh(symbols, probs, color=colors)
            axs[0].set_title('Order Flow Prediction Probabilities by Symbol', fontsize=14)
            axs[0].set_xlabel('Probability')
            axs[0].set_ylabel('Symbol')
            axs[0].axvline(x=0.5, color='black', linestyle='--')
            axs[0].set_xlim(0, 1)
            
            # Add threshold line
            axs[0].axvline(x=self.threshold, color='blue', linestyle='--', label=f'Threshold ({self.threshold})')
            axs[0].legend()
            
            # Add prediction labels
            for i, prob in enumerate(probs):
                direction = 'Bullish' if agg_results[symbols[i]]['direction'] == 'bullish' else 'Bearish'
                confidence = agg_results[symbols[i]]['confidence'] * 100
                axs[0].text(
                    prob + 0.01, 
                    i, 
                    f"{direction} ({confidence:.0f}% conf.)",
                    va='center'
                )
            
            # Plot historical price data if available
            if price_data is not None and len(symbols) > 0:
                # Pick the first symbol for plotting
                symbol_to_plot = symbols[-1]  # Plot the most bullish/bearish symbol
                
                symbol_prices = price_data[price_data['symbol'] == symbol_to_plot].copy()
                if len(symbol_prices) > 5:
                    symbol_prices = symbol_prices.sort_values('date')
                    
                    # Plot the prices
                    axs[1].plot(symbol_prices['date'], symbol_prices['close'], label='Close Price')
                    axs[1].set_title(f'Historical Prices for {symbol_to_plot}', fontsize=14)
                    axs[1].set_xlabel('Date')
                    axs[1].set_ylabel('Price')
                    axs[1].grid(True)
                    
                    # Mark prediction window
                    last_date = symbol_prices['date'].max()
                    prediction_end = last_date + timedelta(days=self.prediction_window)
                    
                    axs[1].axvspan(last_date, prediction_end, alpha=0.2, color='gray')
                    axs[1].text(
                        last_date + (prediction_end - last_date) / 2, 
                        symbol_prices['close'].max(),
                        f"{agg_results[symbol_to_plot]['direction'].upper()}",
                        ha='center',
                        fontsize=12,
                        color='green' if agg_results[symbol_to_plot]['direction'] == 'bullish' else 'red'
                    )
                else:
                    axs[1].text(0.5, 0.5, "Insufficient price data for visualization", 
                              ha='center', va='center', transform=axs[1].transAxes)
            else:
                axs[1].text(0.5, 0.5, "No price data available", 
                          ha='center', va='center', transform=axs[1].transAxes)
            
            plt.tight_layout()
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing predictions: {str(e)}")
            return None, None
    
    def analyze_order_flow_sentiment(self, order_flow_data, window=10):
        """
        Analyze sentiment based on order flow data.
        
        Args:
            order_flow_data (pd.DataFrame): Order flow data
            window (int): Rolling window for sentiment calculation
            
        Returns:
            pd.DataFrame: Sentiment analysis results
        """
        self.logger.info("Analyzing order flow sentiment")
        
        try:
            df = order_flow_data.copy()
            
            # Ensure date is in datetime format
            if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Group by date and symbol
            if 'symbol' in df.columns:
                groupby_cols = ['symbol', pd.Grouper(key='date', freq='D')]
            else:
                groupby_cols = [pd.Grouper(key='date', freq='D')]
            
            # Calculate call and put volumes
            call_volumes = df[df['option_type'] == 'call'].groupby(groupby_cols)['volume'].sum()
            put_volumes = df[df['option_type'] == 'put'].groupby(groupby_cols)['volume'].sum()
            
            # Combine into a DataFrame
            sentiment_df = pd.DataFrame({
                'call_volume': call_volumes,
                'put_volume': put_volumes
            }).reset_index()
            
            # Calculate sentiment ratios
            sentiment_df['call_put_ratio'] = sentiment_df['call_volume'] / sentiment_df['put_volume'].replace(0, 1)
            sentiment_df['put_call_ratio'] = sentiment_df['put_volume'] / sentiment_df['call_volume'].replace(0, 1)
            
            # Calculate log ratios (better for visualization and comparison)
            sentiment_df['log_call_put_ratio'] = np.log(sentiment_df['call_put_ratio'])
            
            # Add rolling averages
            sentiment_df = sentiment_df.sort_values(['symbol', 'date']) if 'symbol' in sentiment_df.columns else sentiment_df.sort_values('date')
            
            if 'symbol' in sentiment_df.columns:
                # Calculate for each symbol
                for symbol in sentiment_df['symbol'].unique():
                    symbol_mask = sentiment_df['symbol'] == symbol
                    sentiment_df.loc[symbol_mask, 'call_put_ratio_ma'] = sentiment_df.loc[symbol_mask, 'call_put_ratio'].rolling(window=window).mean()
                    sentiment_df.loc[symbol_mask, 'log_call_put_ratio_ma'] = sentiment_df.loc[symbol_mask, 'log_call_put_ratio'].rolling(window=window).mean()
            else:
                # Calculate overall
                sentiment_df['call_put_ratio_ma'] = sentiment_df['call_put_ratio'].rolling(window=window).mean()
                sentiment_df['log_call_put_ratio_ma'] = sentiment_df['log_call_put_ratio'].rolling(window=window).mean()
            
            # Calculate z-scores
            sentiment_df['call_put_ratio_z'] = (sentiment_df['call_put_ratio'] - sentiment_df['call_put_ratio_ma']) / sentiment_df['call_put_ratio'].rolling(window=window).std().replace(0, 1)
            
            # Determine sentiment category
            conditions = [
                (sentiment_df['call_put_ratio'] > 2.0),
                (sentiment_df['call_put_ratio'] > 1.5),
                (sentiment_df['call_put_ratio'] > 1.1),
                (sentiment_df['call_put_ratio'] < 0.9),
                (sentiment_df['call_put_ratio'] < 0.67),
                (sentiment_df['call_put_ratio'] < 0.5)
            ]
            
            categories = [
                'Very Bullish',
                'Bullish',
                'Slightly Bullish',
                'Slightly Bearish',
                'Bearish',
                'Very Bearish'
            ]
            
            sentiment_df['sentiment'] = np.select(conditions, categories, default='Neutral')
            
            # Add sentiment score (-1 to 1 scale)
            sentiment_df['sentiment_score'] = np.clip((sentiment_df['call_put_ratio'] - 1) / 2, -1, 1)
            
            self.logger.info(f"Generated sentiment analysis for {len(sentiment_df)} data points")
            
            return sentiment_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow sentiment: {str(e)}")
            return pd.DataFrame()
