"""
Adaptive Classifier Module

This module implements self-supervised learning for market regime classification.
It adapts to changing market conditions and learns from recent market behavior.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class AdaptiveClassifier:
    """
    Self-supervised learning system that adapts to market regimes over time.
    """
    
    # Market regime types
    REGIME_TYPES = ["bullish", "bearish", "volatile", "sideways", "trending"]
    
    def __init__(self, config, tradier_api, feature_extractor, drive_manager=None):
        """
        Initialize the AdaptiveClassifier.
        
        Args:
            config (dict): Configuration settings
            tradier_api: Instance of TradierAPI for market data
            feature_extractor: RegimeFeatureExtractor instance
            drive_manager: Optional Google Drive Manager for saving/loading models
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.feature_extractor = feature_extractor
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.classifier_config = config.get("market_regime", {}).get("adaptive_classifier", {})
        
        # Model parameters
        self.n_estimators = self.classifier_config.get("n_estimators", 100)
        self.max_depth = self.classifier_config.get("max_depth", 10)
        self.min_samples_leaf = self.classifier_config.get("min_samples_leaf", 5)
        
        # Initialize model
        self.model = None
        self.last_training_date = None
        self.feature_importance = {}
        
        # Load or create model
        self._initialize_model()
        
        # Training data history
        self.training_data = []
        
        # Create directories for models
        os.makedirs("data/market_regimes", exist_ok=True)
        
        self.logger.info("AdaptiveClassifier initialized")
    
    def _initialize_model(self):
        """Initialize or load the classification model."""
        model_file = "data/market_regimes/adaptive_classifier.joblib"
        
        try:
            # Try to load existing model
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                self.logger.info("Loaded existing adaptive classification model")
                
                # Load last training date if available
                metadata_file = "data/market_regimes/classifier_metadata.json"
                if os.path.exists(metadata_file):
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if 'last_training_date' in metadata:
                            self.last_training_date = datetime.strptime(
                                metadata['last_training_date'], '%Y-%m-%d'
                            ).date()
                
            elif self.drive_manager and self.drive_manager.file_exists("adaptive_classifier.joblib"):
                # Try to load from Google Drive
                model_data = self.drive_manager.download_file_binary("adaptive_classifier.joblib")
                
                # Save locally
                with open(model_file, "wb") as f:
                    f.write(model_data)
                
                self.model = joblib.load(model_file)
                self.logger.info("Loaded adaptive classification model from Google Drive")
                
                # Load metadata if available
                if self.drive_manager.file_exists("classifier_metadata.json"):
                    metadata_content = self.drive_manager.download_file("classifier_metadata.json")
                    import json
                    metadata = json.loads(metadata_content)
                    if 'last_training_date' in metadata:
                        self.last_training_date = datetime.strptime(
                            metadata['last_training_date'], '%Y-%m-%d'
                        ).date()
            else:
                # Create new model
                self._create_new_model()
                
        except Exception as e:
            self.logger.error(f"Error initializing adaptive model: {str(e)}")
            # Fall back to a new model
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new classification model."""
        self.logger.info("Creating new adaptive classification model")
        
        # Create pipeline with preprocessing and classifier
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42
            ))
        ])
        
        # Not trained yet
        self.last_training_date = None
    
    def needs_training(self):
        """
        Check if the model needs to be retrained.
        
        Returns:
            bool: True if retraining is needed
        """
        # If never trained, definitely needs training
        if self.last_training_date is None:
            return True
        
        # Check time since last training
        today = datetime.now().date()
        training_interval = self.classifier_config.get("training_interval_days", 7)
        
        return (today - self.last_training_date).days >= training_interval
    
    def collect_training_data(self, symbols=None, lookback_days=90):
        """
        Collect training data for adaptive learning.
        
        Args:
            symbols (list): List of symbols to use for training
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Collected training data
        """
        try:
            if symbols is None:
                symbols = self.classifier_config.get("training_symbols", ["SPY", "QQQ", "IWM", "DIA"])
            
            self.logger.info(f"Collecting training data from {len(symbols)} symbols")
            
            all_data = []
            
            for symbol in symbols:
                # Get historical data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                
                historical_data = self.tradier_api.get_historical_data(
                    symbol, interval='daily', start_date=start_date, end_date=end_date
                )
                
                if historical_data.empty:
                    self.logger.warning(f"Could not get historical data for {symbol}")
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_features(historical_data)
                
                if features.empty:
                    continue
                
                # Auto-generate labels using rules
                labels = self._generate_labels(features, historical_data)
                
                # Add labels to features
                features['regime'] = labels
                
                # Add symbol
                features['symbol'] = symbol
                
                all_data.append(features)
            
            if not all_data:
                self.logger.warning("No training data collected")
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Collected {len(combined_data)} training samples")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_labels(self, features, historical_data):
        """
        Generate regime labels using rule-based approach.
        
        Args:
            features (pd.DataFrame): Feature data
            historical_data (pd.DataFrame): Historical price data
            
        Returns:
            list: Generated labels
        """
        # Initialize labels
        labels = []
        
        for idx, row in features.iterrows():
            # Skip if not enough data yet
            if idx < 5:
                labels.append("sideways")  # Default label
                continue
            
            # Get feature values
            price_change = row.get('price_change_pct', 0)
            volatility = row.get('volatility', 0)
            trend_strength = row.get('trend_strength', 0)
            rsi = row.get('rsi', 50)
            adx = row.get('adx', 0)
            
            # Rule-based labeling
            if volatility > 2.0 or row.get('volatility_ratio', 1) > 1.5:
                label = "volatile"
            elif trend_strength > 0.7 and adx > 25:
                if price_change > 0:
                    label = "bullish"
                else:
                    label = "bearish"
            elif abs(price_change) < 0.8 and volatility < 1.2:
                label = "sideways"
            elif trend_strength > 0.5:
                label = "trending"
            elif price_change > 1.0 and rsi > 50:
                label = "bullish"
            elif price_change < -1.0 and rsi < 50:
                label = "bearish"
            else:
                label = "sideways"  # Default
            
            labels.append(label)
        
        return labels
    
    def train_model(self, training_data=None):
        """
        Train the adaptive classification model.
        
        Args:
            training_data (pd.DataFrame, optional): Training data
            
        Returns:
            bool: True if training was successful
        """
        try:
            # Collect training data if not provided
            if training_data is None or training_data.empty:
                training_data = self.collect_training_data()
            
            if training_data.empty:
                self.logger.warning("No training data available")
                return False
            
            self.logger.info(f"Training adaptive classifier with {len(training_data)} samples")
            
            # Prepare features and target
            X = training_data.drop(['regime', 'date', 'symbol'], axis=1, errors='ignore')
            y = training_data['regime']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            self.logger.info(f"Model trained with test accuracy: {accuracy:.4f}")
            
            # Save feature importance
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                importances = self.model.named_steps['classifier'].feature_importances_
                feature_importance = dict(zip(X.columns, importances))
                self.feature_importance = dict(sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
            
            # Save model
            joblib.dump(self.model, "data/market_regimes/adaptive_classifier.joblib")
            
            # Update last training date
            self.last_training_date = datetime.now().date()
            
            # Save metadata
            import json
            metadata = {
                'last_training_date': self.last_training_date.strftime('%Y-%m-%d'),
                'accuracy': accuracy,
                'n_samples': len(training_data),
                'feature_importance': self.feature_importance
            }
            
            with open("data/market_regimes/classifier_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open("data/market_regimes/adaptive_classifier.joblib", "rb") as f:
                    self.drive_manager.upload_file(
                        "adaptive_classifier.joblib", 
                        f.read(),
                        folder="market_regimes",
                        mime_type="application/octet-stream"
                    )
                
                self.drive_manager.upload_file(
                    "classifier_metadata.json",
                    json.dumps(metadata, indent=2),
                    folder="market_regimes",
                    mime_type="application/json"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training adaptive model: {str(e)}")
            return False
    
    def predict_regime(self, features):
        """
        Predict market regime using the adaptive model.
        
        Args:
            features (pd.DataFrame): Feature data (single row)
            
        Returns:
            str: Predicted market regime
        """
        try:
            if self.model is None:
                self.logger.warning("No model available for prediction")
                return "unknown"
            
            # Prepare features
            X = features.drop(['date', 'symbol'], axis=1, errors='ignore')
            
            # Make prediction
            regime = self.model.predict(X)[0]
            
            # Get prediction probability
            probabilities = self.model.predict_proba(X)[0]
            
            # Get index of predicted class
            predicted_idx = list(self.model.classes_).index(regime)
            confidence = probabilities[predicted_idx]
            
            self.logger.debug(f"Predicted regime: {regime} (confidence: {confidence:.4f})")
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error predicting regime: {str(e)}")
            return "unknown"
    
    def get_regime_probability(self, features):
        """
        Get probability distribution over all regimes.
        
        Args:
            features (pd.DataFrame): Feature data (single row)
            
        Returns:
            dict: Regime probabilities
        """
        try:
            if self.model is None:
                return {regime: 0.2 for regime in self.REGIME_TYPES}
            
            # Prepare features
            X = features.drop(['date', 'symbol'], axis=1, errors='ignore')
            
            # Get probabilities
            probabilities = self.model.predict_proba(X)[0]
            
            # Map to regime types
            regime_probs = {}
            for i, regime in enumerate(self.model.classes_):
                regime_probs[regime] = probabilities[i]
            
            return regime_probs
            
        except Exception as e:
            self.logger.error(f"Error getting regime probabilities: {str(e)}")
            return {regime: 0.2 for regime in self.REGIME_TYPES}
    
    def get_feature_importance(self):
        """
        Get feature importance ranking.
        
        Returns:
            dict: Feature importance scores
        """
        return self.feature_importance
    
    def adaptive_update(self, market_data, observed_regime=None):
        """
        Update the model adaptively based on recent market data.
        
        Args:
            market_data (pd.DataFrame): Recent market data
            observed_regime (str, optional): Manually observed regime
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(market_data)
            
            if features.empty:
                return False
            
            # Generate labels if not provided
            if observed_regime is None:
                labels = self._generate_labels(features, market_data)
            else:
                # Use provided label for all rows
                labels = [observed_regime] * len(features)
            
            # Add labels to features
            features['regime'] = labels
            
            # Add to training data history
            self.training_data.append(features)
            
            # Limit history size
            max_history = self.classifier_config.get("max_history_size", 10)
            if len(self.training_data) > max_history:
                self.training_data = self.training_data[-max_history:]
            
            # Check if retraining is needed
            if self.needs_training() and len(self.training_data) >= 3:
                # Combine all available training data
                combined_data = pd.concat(self.training_data, ignore_index=True)
                
                # Retrain model
                self.train_model(combined_data)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in adaptive update: {str(e)}")
            return False
