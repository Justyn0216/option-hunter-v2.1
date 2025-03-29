"""
Market State Tracker Module

This module detects market regimes and tracks transitions between different market states.
It analyzes market data to determine whether the market is in a bullish, bearish, 
volatile, or sideways regime.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from sklearn.cluster import KMeans
import joblib

from src.market_regime.regime_feature_extractor import RegimeFeatureExtractor

class MarketStateTracker:
    """
    Tracks and classifies market regimes based on various indicators and features.
    """
    
    # Market regime types
    BULLISH = "bullish"
    BEARISH = "bearish"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    TRENDING = "trending"
    UNKNOWN = "unknown"
    
    def __init__(self, config, tradier_api, drive_manager=None):
        """
        Initialize the MarketStateTracker component.
        
        Args:
            config (dict): Configuration settings
            tradier_api: Instance of TradierAPI for market data
            drive_manager: Optional Google Drive Manager for saving/loading models
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.regime_config = config.get("market_regime", {})
        
        # Initialize feature extractor
        self.feature_extractor = RegimeFeatureExtractor(tradier_api)
        
        # Current regime information
        self.current_regime = self.UNKNOWN
        self.regime_start_time = datetime.now()
        self.regime_history = []
        
        # Model for regime classification
        self.model = None
        
        # Load or initialize model
        self._initialize_model()
        
        # Create directory for regime logs
        os.makedirs("logs/regime_transitions", exist_ok=True)
        
        self.logger.info("MarketStateTracker initialized")
    
    def _initialize_model(self):
        """Initialize or load the regime classification model."""
        model_file = "data/market_regimes/regime_classifier.joblib"
        
        try:
            # Try to load existing model
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                self.logger.info("Loaded existing regime classification model")
            elif self.drive_manager and self.drive_manager.file_exists("regime_classifier.joblib"):
                # Try to load from Google Drive
                model_data = self.drive_manager.download_file_binary("regime_classifier.joblib")
                
                # Save locally
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                with open(model_file, "wb") as f:
                    f.write(model_data)
                
                self.model = joblib.load(model_file)
                self.logger.info("Loaded regime classification model from Google Drive")
            else:
                # Initialize new model if none exists
                self._train_initial_model()
                
        except Exception as e:
            self.logger.error(f"Error initializing regime model: {str(e)}")
            # Fall back to a basic model
            self._train_initial_model()
    
    def _train_initial_model(self):
        """Train an initial regime classification model using K-means clustering."""
        try:
            self.logger.info("Training initial regime classification model")
            
            # Use SPY as market proxy
            features = self._get_historical_features("SPY")
            
            if features.empty:
                self.logger.warning("Could not get features for initial model training")
                self.model = None
                return
            
            # Use K-means for unsupervised regime clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            
            # Select numeric columns for clustering
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            scaled_features = (features[numeric_cols] - features[numeric_cols].mean()) / features[numeric_cols].std()
            
            # Remove NaN values
            scaled_features = scaled_features.fillna(0)
            
            # Fit model
            kmeans.fit(scaled_features)
            
            # Save model
            os.makedirs("data/market_regimes", exist_ok=True)
            joblib.dump(kmeans, "data/market_regimes/regime_classifier.joblib")
            
            self.model = kmeans
            self.logger.info("Initial regime classification model trained")
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open("data/market_regimes/regime_classifier.joblib", "rb") as f:
                    self.drive_manager.upload_file(
                        "regime_classifier.joblib", 
                        f.read(),
                        mime_type="application/octet-stream"
                    )
                self.logger.info("Uploaded regime model to Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error training initial model: {str(e)}")
            self.model = None
    
    def _get_historical_features(self, symbol, days=60):
        """
        Get historical features for a symbol.
        
        Args:
            symbol (str): Symbol to get features for
            days (int): Number of days of history
            
        Returns:
            pd.DataFrame: Features dataframe
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Get historical price data
            historical_data = self.tradier_api.get_historical_data(
                symbol, interval='daily', start_date=start_date, end_date=end_date
            )
            
            if historical_data.empty:
                return pd.DataFrame()
                
            # Get features using the feature extractor
            features = self.feature_extractor.extract_features(historical_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error getting historical features: {str(e)}")
            return pd.DataFrame()
